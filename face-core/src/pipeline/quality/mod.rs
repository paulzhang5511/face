// ====================================================================
// 人脸图像质量评估 — eDifFIQA
// ====================================================================
//
// 预处理: 5点对齐 112x112 -> RGB (x-127.5)/127.5 -> NCHW
// 后处理: 输出 squeeze 后直接作为分数 (模型已内含 sigmoid)

use anyhow::Result;
use image::DynamicImage;
use log::info;
use ndarray::Array2;
use ort::value::TensorRef;
use rayon::prelude::*;

use crate::inference::InferenceSession;
use crate::types::QualityResult;
use crate::utils::alignment::estimate_norm_112;
use crate::utils::image_utils::warp_affine_bilinear;

use crate::pipeline::recognition::arcface_blob;

/// eDifFIQA 人脸质量评估器
pub struct EDifFIQA {
    session: InferenceSession,
    input_size: u32,
}

impl EDifFIQA {
    /// 从推理会话创建质量评估器 (输入 112x112)
    pub fn from_session(session: InferenceSession) -> Result<Self> {
        let input_size = 112u32;
        info!(
            "EDifFIQA: input_size={}x{}, normalization=(x-127.5)/127.5",
            input_size, input_size
        );
        Ok(Self {
            session,
            input_size,
        })
    }

    /// 端到端质量评估: 5点对齐 -> 预处理 -> 推理 -> 分数
    pub fn predict(
        &mut self,
        image: &DynamicImage,
        landmarks: &Array2<f32>,
    ) -> Result<QualityResult> {
        let shape = landmarks.shape();
        if shape != [5, 2] {
            anyhow::bail!(
                "EDifFIQA requires 5 landmarks of shape (5, 2), got {:?}",
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
        let rgb = image.to_rgb8();
        let aligned = warp_affine_bilinear(&rgb, &forward, self.input_size, self.input_size);

        let input_array = arcface_blob(&aligned, self.input_size);
        let input = TensorRef::from_array_view(&input_array)?;
        let output_name = self.session.session().outputs()[0].name().to_string();
        let outputs = self.session.session_mut().run(ort::inputs![input])?;

        // 提取输出数据到 owned Vec (避免借用冲突)
        let (_shape, out_ref) = outputs
            .get(output_name.as_str())
            .ok_or_else(|| anyhow::anyhow!("eDifFIQA model output '{}' not found", output_name))?
            .try_extract_tensor::<f32>()?;
        let out_data: Vec<f32> = out_ref.to_vec();
        drop(outputs);

        Ok(QualityResult {
            score: parse_quality_score(&out_data),
        })
    }
    /// 批量质量评估: 逐脸对齐 + 一次 batch 推理
    ///
    /// 输入按顺序对应输出; 无 landmarks 的脸对应 None
    pub fn predict_batch(
        &mut self,
        image: &DynamicImage,
        faces: &[&crate::types::DetectedFace],
    ) -> Result<Vec<Option<QualityResult>>> {
        if faces.is_empty() {
            return Ok(Vec::new());
        }
        let size = self.input_size;
        let rgb = image.to_rgb8();
        // 对齐互相独立, rayon 并行
        let prepped: Vec<Option<ndarray::Array4<f32>>> = faces
            .par_iter()
            .map(|face| {
                let landmarks = face.landmarks.as_ref()?;
                let pts: [(f32, f32); 5] = [
                    (landmarks[[0, 0]], landmarks[[0, 1]]),
                    (landmarks[[1, 0]], landmarks[[1, 1]]),
                    (landmarks[[2, 0]], landmarks[[2, 1]]),
                    (landmarks[[3, 0]], landmarks[[3, 1]]),
                    (landmarks[[4, 0]], landmarks[[4, 1]]),
                ];
                let (forward, _inverse) = estimate_norm_112(&pts);
                let aligned = warp_affine_bilinear(&rgb, &forward, size, size);
                let mut blob = ndarray::Array4::<f32>::zeros((1, 3, size as usize, size as usize));
                for (x, y, pixel) in aligned.enumerate_pixels() {
                    for ch in 0..3 {
                        blob[[0, ch, y as usize, x as usize]] = (pixel[ch] as f32 - 127.5) / 127.5;
                    }
                }
                Some(blob)
            })
            .collect();
        let mut inputs = Vec::with_capacity(faces.len());
        let mut valid = Vec::with_capacity(faces.len());
        for (i, r) in prepped.into_iter().enumerate() {
            if let Some(blob) = r {
                inputs.push(blob);
                valid.push(i);
            }
        }

        let mut results: Vec<Option<QualityResult>> = vec![None; faces.len()];
        if inputs.is_empty() {
            return Ok(results);
        }

        // 拼接为 (N, 3, H, W)
        let n = inputs.len();
        let mut batch = ndarray::Array4::<f32>::zeros((n, 3, size as usize, size as usize));
        for (b, blob) in inputs.iter().enumerate() {
            batch
                .slice_mut(ndarray::s![b..b + 1, .., .., ..])
                .assign(blob);
        }

        let tensor = TensorRef::from_array_view(&batch)?;
        let output_name = self.session.session().outputs()[0].name().to_string();
        let outputs = self.session.session_mut().run(ort::inputs![tensor])?;
        let (_shape, out_ref) = outputs
            .get(output_name.as_str())
            .ok_or_else(|| anyhow::anyhow!("eDifFIQA model output '{}' not found", output_name))?
            .try_extract_tensor::<f32>()?;
        let out_data: Vec<f32> = out_ref.to_vec();
        drop(outputs);

        let dim = if out_data.is_empty() {
            1
        } else {
            out_data.len() / n
        };
        for (b, &i) in valid.iter().enumerate() {
            let row = &out_data[b * dim..(b + 1) * dim];
            results[i] = Some(QualityResult {
                score: parse_quality_score(row),
            });
        }
        Ok(results)
    }
}

/// 输出解析: 原始输出 (任意形状) squeeze 后取第一个元素作为分数
pub fn parse_quality_score(output: &[f32]) -> f32 {
    if output.is_empty() { 0.0 } else { output[0] }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_quality_score_scalar() {
        // 单值输出
        assert!((parse_quality_score(&[0.8734]) - 0.8734).abs() < 1e-6);
    }

    #[test]
    fn test_parse_quality_score_batch() {
        // 批量输出取第一个元素
        assert!((parse_quality_score(&[0.25, 0.9, 0.1]) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_parse_quality_score_empty() {
        // 空输出按 0 处理
        assert_eq!(parse_quality_score(&[]), 0.0);
    }
}
