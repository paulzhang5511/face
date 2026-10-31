// ====================================================================
// 人脸识别 — 512维特征嵌入
// ====================================================================
//
// 预处理:
//   1. 5点相似变换对齐: landmarks (5,2) -> 112x112 (标准参考坐标, forward matrix warp)
//   2. RGB, (pixel - 127.5) / 127.5 -> [-1, 1]
//   3. NCHW float32
// 注: 识别模型输入为 RGB; Rust 侧 DynamicImage 本身即 RGB, 无需交换通道

use anyhow::Result;
use image::{DynamicImage, RgbImage};
use log::info;
use ndarray::{Array2, Array4};
use ort::value::TensorRef;
use rayon::prelude::*;

use crate::inference::InferenceSession;
use crate::types::DetectedFace;
use crate::utils::alignment::estimate_norm_112;
use crate::utils::image_utils::warp_affine_bilinear;

/// 人脸识别器 (112x112 输入, 输出 512维特征嵌入)
///
/// 主流 112x112 识别模型预处理配置一致
/// (mean=127.5 / std=127.5 / 112x112, 输出 512维),
/// 因此同一结构可通过 `from_session` 加载任意 112x112 识别模型
pub struct ArcFace {
    session: InferenceSession,
    input_size: u32,
}

impl ArcFace {
    /// 从推理会话创建识别器 (输入 112x112)
    pub fn from_session(session: InferenceSession) -> Result<Self> {
        let input_size = 112u32;
        info!(
            "ArcFace: input_size={}x{}, normalization=(x-127.5)/127.5",
            input_size, input_size
        );
        Ok(Self {
            session,
            input_size,
        })
    }

    /// 提取人脸特征嵌入 (原始输出, 未归一化)
    ///
    /// 流程: 5点对齐 -> 112x112 warp -> RGB [-1,1] -> NCHW -> 推理
    pub fn get_embedding(
        &mut self,
        image: &DynamicImage,
        landmarks: &Array2<f32>,
    ) -> Result<Vec<f32>> {
        let aligned = align_face(image, landmarks, self.input_size)?;
        let input_array = arcface_blob(&aligned, self.input_size);

        let input = TensorRef::from_array_view(&input_array)?;
        let output_name = self.session.session().outputs()[0].name().to_string();
        let outputs = self.session.session_mut().run(ort::inputs![input])?;

        // 提取输出数据到 owned Vec (避免借用冲突)
        let (_shape, emb_ref) = outputs
            .get(output_name.as_str())
            .ok_or_else(|| anyhow::anyhow!("Recognition model output '{}' not found", output_name))?
            .try_extract_tensor::<f32>()?;
        let embedding: Vec<f32> = emb_ref.to_vec();
        drop(outputs);

        Ok(embedding)
    }

    /// 提取 L2 归一化后的人脸特征嵌入
    pub fn get_normalized_embedding(
        &mut self,
        image: &DynamicImage,
        landmarks: &Array2<f32>,
    ) -> Result<Vec<f32>> {
        let embedding = self.get_embedding(image, landmarks)?;
        Ok(l2_normalize(&embedding))
    }

    /// 批量提取 L2 归一化特征嵌入 (batch 一次推理)
    ///
    /// 输入按顺序对应输出; 单脸对齐失败时对应输出为 None
    pub fn get_normalized_embeddings_batch(
        &mut self,
        image: &DynamicImage,
        faces: &[&DetectedFace],
    ) -> Result<Vec<Option<Vec<f32>>>> {
        if faces.is_empty() {
            return Ok(Vec::new());
        }
        let size = self.input_size as usize;
        // 单次整图转 RGB, 逐脸对齐后拼成 (N, 3, 112, 112)
        // 对齐互相独立, rayon 并行
        let rgb = image.to_rgb8();
        let prepped: Vec<Result<RgbImage>> = faces
            .par_iter()
            .map(|face| {
                let landmarks = face
                    .landmarks
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("face has no landmarks"))?;
                align_face_rgb(&rgb, landmarks, self.input_size)
            })
            .collect();
        let mut aligned_all = Vec::with_capacity(faces.len());
        let mut valid = Vec::with_capacity(faces.len());
        for (i, r) in prepped.into_iter().enumerate() {
            match r {
                Ok(aligned) => {
                    aligned_all.push(aligned);
                    valid.push(i);
                }
                Err(e) => {
                    log::debug!("ArcFace: face {} alignment failed: {}", i, e);
                }
            }
        }

        let mut results: Vec<Option<Vec<f32>>> = vec![None; faces.len()];
        if aligned_all.is_empty() {
            return Ok(results);
        }

        let n = aligned_all.len();
        let mut input = ndarray::Array4::<f32>::zeros((n, 3, size, size));
        for (b, aligned) in aligned_all.iter().enumerate() {
            for (x, y, pixel) in aligned.enumerate_pixels() {
                for ch in 0..3 {
                    input[[b, ch, y as usize, x as usize]] = (pixel[ch] as f32 - 127.5) / 127.5;
                }
            }
        }

        let tensor = TensorRef::from_array_view(&input)?;
        let output_name = self.session.session().outputs()[0].name().to_string();
        let outputs = self.session.session_mut().run(ort::inputs![tensor])?;
        let (_shape, emb_ref) = outputs
            .get(output_name.as_str())
            .ok_or_else(|| anyhow::anyhow!("Recognition model output '{}' not found", output_name))?
            .try_extract_tensor::<f32>()?;

        let emb = emb_ref.to_vec();
        let dim = if emb.is_empty() { 0 } else { emb.len() / n };
        for (b, &i) in valid.iter().enumerate() {
            let row = &emb[b * dim..(b + 1) * dim];
            results[i] = Some(l2_normalize(row));
        }
        Ok(results)
    }
}

/// L2 归一化 (对齐 base.py: 范数为 0 时原样返回)
pub fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

/// 余弦相似度 (epsilon 1e-5 防止除零)
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    // 小 epsilon 防止除零
    dot / (norm_a * norm_b + 1e-5)
}

/// 5点对齐裁剪: landmarks (5,2) -> size x size RGB 图像
///
/// 关键点顺序: 左眼, 右眼, 鼻, 左嘴角, 右嘴角 (与 DetectedFace.landmarks 一致)
/// warp 使用 forward matrix (正向仿射矩阵逐像素采样)
pub fn align_face(image: &DynamicImage, landmarks: &Array2<f32>, size: u32) -> Result<RgbImage> {
    let rgb = image.to_rgb8();
    align_face_rgb(&rgb, landmarks, size)
}

/// 同 align_face, 但接收已转换的 RGB 图像 (批量路径复用, 避免重复整图转换)
pub fn align_face_rgb(rgb: &RgbImage, landmarks: &Array2<f32>, size: u32) -> Result<RgbImage> {
    let shape = landmarks.shape();
    if shape != [5, 2] {
        anyhow::bail!(
            "align_face requires 5 landmarks of shape (5, 2), got {:?}",
            shape
        );
    }
    let pts: [(f32, f32); 5] = [
        (landmarks[[0, 0]], landmarks[[0, 1]]),
        (landmarks[[1, 0]], landmarks[[1, 1]]),
        (landmarks[[2, 0]], landmarks[[2, 1]]),
        (landmarks[[3, 0]], landmarks[[3, 1]]),
        (landmarks[[4, 0]], landmarks[[4, 1]]),
    ];
    let (forward, _inverse) = estimate_norm_112(&pts);
    Ok(warp_affine_bilinear(rgb, &forward, size, size))
}

/// 对齐后人脸 -> NCHW blob, (x - 127.5) / 127.5 归一化
pub(crate) fn arcface_blob(aligned: &RgbImage, size: u32) -> Array4<f32> {
    let mut arr = Array4::<f32>::zeros((1, 3, size as usize, size as usize));
    for (x, y, pixel) in aligned.enumerate_pixels() {
        let c = &pixel.0;
        for ch in 0..3 {
            arr[[0, ch, y as usize, x as usize]] = (c[ch] as f32 - 127.5) / 127.5;
        }
    }
    arr
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_same_vector() {
        // 同向量 (非单位) 应得到 ~1 (epsilon 使其略小于 1)
        let a = vec![3.0, 4.0, 0.0];
        let s = cosine_similarity(&a, &a);
        assert!((s - 1.0).abs() < 1e-4, "got {}", s);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        // 正交向量应为 0
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let s = cosine_similarity(&a, &b);
        assert!(s.abs() < 1e-6, "got {}", s);
    }

    #[test]
    fn test_cosine_similarity_known_value() {
        // cos(45°) ~ 0.7071: dot=1, |a|=1, |b|=sqrt(2)
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 1.0];
        let s = cosine_similarity(&a, &b);
        let expected = 1.0 / (1.0f32 * 2.0f32.sqrt() + 1e-5);
        assert!((s - expected).abs() < 1e-6, "got {}", s);
    }

    #[test]
    fn test_cosine_similarity_zero_vectors() {
        // 零向量: epsilon 防止除零, 结果为 0
        let a = vec![0.0f32, 0.0, 0.0];
        let s = cosine_similarity(&a, &a);
        assert!(s.abs() < 1e-6, "got {}", s);
    }

    #[test]
    fn test_l2_normalize() {
        let v = vec![3.0, 4.0];
        let n = l2_normalize(&v);
        assert!((n[0] - 0.6).abs() < 1e-6);
        assert!((n[1] - 0.8).abs() < 1e-6);
        // 范数应为 1
        let norm: f32 = n.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_normalize_zero_vector() {
        // 零向量原样返回
        let v = vec![0.0f32, 0.0];
        let n = l2_normalize(&v);
        assert_eq!(n, vec![0.0f32, 0.0]);
    }

    #[test]
    fn test_arcface_blob_normalization() {
        // 中灰 128 -> (128-127.5)/127.5 ~ 0.0039; 黑 0 -> -1; 白 255 -> ~1.0039 (f32 精度)
        let mut img = RgbImage::new(2, 2);
        img.put_pixel(0, 0, image::Rgb([0u8, 0, 0]));
        img.put_pixel(1, 0, image::Rgb([255u8, 255, 255]));
        img.put_pixel(0, 1, image::Rgb([255u8, 255, 255]));
        img.put_pixel(1, 1, image::Rgb([0u8, 0, 0]));

        let blob = arcface_blob(&img, 2);
        assert_eq!(blob.shape(), &[1, 3, 2, 2]);
        for ch in 0..3 {
            assert!((blob[[0, ch, 0, 0]] + 1.0).abs() < 1e-6);
            assert!((blob[[0, ch, 0, 1]] - 1.0).abs() < 1e-2);
            assert!((blob[[0, ch, 1, 0]] - 1.0).abs() < 1e-2);
            assert!((blob[[0, ch, 1, 1]] + 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_align_face_rejects_bad_shape() {
        let img = DynamicImage::new_rgb8(64, 64);
        let bad = Array2::<f32>::zeros((5, 3));
        assert!(align_face(&img, &bad, 112).is_err());
    }

    #[test]
    fn test_align_face_output_size() {
        // 输出应为 112x112
        let img = DynamicImage::new_rgb8(128, 128);
        let mut lm = Array2::<f32>::zeros((5, 2));
        for (i, &(x, y)) in crate::utils::alignment::ARC_FACE_REFERENCE_112
            .iter()
            .enumerate()
        {
            lm[[i, 0]] = x;
            lm[[i, 1]] = y;
        }
        let aligned = align_face(&img, &lm, 112).unwrap();
        assert_eq!(aligned.dimensions(), (112, 112));
    }
}
