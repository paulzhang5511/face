// ====================================================================
// FaceMesh — 468点3D密集人脸网格
// ====================================================================
//
// 模型: Face Mesh V1
//       - 输入: 192x192 RGB, 归一化到 [0, 1]
//       - 输出: (N, 468, 3) 关键点 + (N, 1) 存在得分
//       - 预处理: ROI 方形裁剪 (1.5x bbox), 基于眼角的旋转校正

use anyhow::Result;
use image::{DynamicImage, GenericImageView};
use log::{info, trace};
use ndarray::Array2;
use ort::value::TensorRef;

use crate::inference::InferenceSession;
use crate::types::{BBox, DetectedFace, FaceMeshResult};
use crate::utils::alignment::invert_affine;

/// FaceMesh 人脸网格模型
pub struct FaceMesh {
    session: InferenceSession,
    input_size: u32,
    roi_margin: f32,
}

impl FaceMesh {
    /// 创建 FaceMesh 模型
    pub fn from_session(session: InferenceSession, roi_margin: f32) -> Result<Self> {
        let input_size = 192u32;
        info!(
            "FaceMesh: input_size={}x{}, roi_margin={}",
            input_size, input_size, roi_margin
        );
        Ok(Self {
            session,
            input_size,
            roi_margin,
        })
    }

    /// 对单个检测到的人脸预测 3D 网格
    pub fn predict(&mut self, image: &DynamicImage, face: &DetectedFace) -> Result<FaceMeshResult> {
        let roi = self.roi_from_box(&face.bbox, face.landmarks.as_ref());

        let (input_array, inv_matrix) = self.preprocess(image, &roi)?;

        let input = TensorRef::from_array_view(&input_array)?;
        let outputs = self.session.session_mut().run(ort::inputs![input])?;

        // 提取输出数据到 owned Vec (输出名缺失返回 Err 而非 panic)
        let landmarks_ref = outputs
            .get("landmarks")
            .ok_or_else(|| anyhow::anyhow!("FaceMesh model output 'landmarks' not found"))?
            .try_extract_tensor::<f32>()?;
        let logits_ref = outputs
            .get("score")
            .ok_or_else(|| anyhow::anyhow!("FaceMesh model output 'score' not found"))?
            .try_extract_tensor::<f32>()?;
        let (_lm_shape, landmarks_ref) = landmarks_ref;
        let (_log_shape, logits_ref) = logits_ref;
        let landmarks_data: Vec<f32> = landmarks_ref.to_vec();
        let logits_data: Vec<f32> = logits_ref.to_vec();
        drop(outputs);

        let result = Self::postprocess(&landmarks_data, &logits_data, &inv_matrix)?;

        trace!(
            "FaceMesh: predicted 468 landmarks, score={:.4}",
            result.score
        );
        Ok(result)
    }

    /// 批量预测多个人脸的 3D 网格
    ///
    /// 单次整图 RGB 转换 + 一次 batch 推理 (模型 batch 维动态)
    pub fn predict_batch(
        &mut self,
        image: &DynamicImage,
        faces: &[DetectedFace],
    ) -> Result<Vec<FaceMeshResult>> {
        if faces.is_empty() {
            return Ok(Vec::new());
        }
        let size = self.input_size as usize;
        let rgb = image.to_rgb8();
        let (img_w, img_h) = image.dimensions();

        // 逐脸 ROI + 仿射预处理, 拼成 (N, 3, size, size)
        let mut blobs = Vec::with_capacity(faces.len());
        let mut invs = Vec::with_capacity(faces.len());
        for (i, face) in faces.iter().enumerate() {
            let roi = self.roi_from_box(&face.bbox, face.landmarks.as_ref());
            match Self::preprocess_rgb(&rgb, img_w, img_h, &roi, self.input_size) {
                Ok((arr, inv)) => {
                    blobs.push(arr);
                    invs.push(inv);
                }
                Err(e) => {
                    log::debug!("FaceMesh: face {} preprocessing failed: {}", i, e);
                }
            }
        }

        let mut results = Vec::with_capacity(faces.len());
        if blobs.is_empty() {
            info!("FaceMesh: batch predicted 0/{} faces", faces.len());
            return Ok(results);
        }

        let n = blobs.len();
        let mut batch = ndarray::Array4::<f32>::zeros((n, 3, size, size));
        for (b, blob) in blobs.iter().enumerate() {
            batch
                .slice_mut(ndarray::s![b..b + 1, .., .., ..])
                .assign(blob);
        }

        let tensor = TensorRef::from_array_view(&batch)?;
        let outputs = self.session.session_mut().run(ort::inputs![tensor])?;
        let landmarks_ref = outputs
            .get("landmarks")
            .ok_or_else(|| anyhow::anyhow!("FaceMesh model output 'landmarks' not found"))?
            .try_extract_tensor::<f32>()?;
        let logits_ref = outputs
            .get("score")
            .ok_or_else(|| anyhow::anyhow!("FaceMesh model output 'score' not found"))?
            .try_extract_tensor::<f32>()?;
        let (_lm_shape, landmarks_ref) = landmarks_ref;
        let (_log_shape, logits_ref) = logits_ref;
        let landmarks_data: Vec<f32> = landmarks_ref.to_vec();
        let logits_data: Vec<f32> = logits_ref.to_vec();
        drop(outputs);

        // 每脸输出 (1, 1404) landmarks + (1, 1) score, 按行切分后逐脸反变换
        let lm_per_face = landmarks_data.len() / n;
        let log_per_face = if logits_data.is_empty() {
            0
        } else {
            logits_data.len() / n
        };
        for b in 0..n {
            let lm = &landmarks_data[b * lm_per_face..(b + 1) * lm_per_face];
            let lg = if log_per_face > 0 {
                &logits_data[b * log_per_face..(b + 1) * log_per_face]
            } else {
                &[]
            };
            match Self::postprocess(lm, lg, &invs[b]) {
                Ok(r) => results.push(r),
                Err(e) => {
                    log::debug!("FaceMesh: face {} postprocess failed: {}", b, e);
                }
            }
        }
        info!(
            "FaceMesh: batch predicted {}/{} faces",
            results.len(),
            faces.len()
        );
        Ok(results)
    }

    /// 从 bbox 计算 ROI (正方形区域)
    fn roi_from_box(&self, bbox: &BBox, keypoints: Option<&Array2<f32>>) -> (f32, f32, f32, f32) {
        let w = bbox.width();
        let h = bbox.height();
        let side = (1.0 + 2.0 * self.roi_margin) * w.max(h);
        let (cx, cy) = bbox.center();

        let mut angle = 0.0f32;
        if let Some(kps) = keypoints {
            if kps.shape()[0] >= 2 {
                let (ex0, ey0) = (kps[[0, 0]], kps[[0, 1]]);
                let (ex1, ey1) = (kps[[1, 0]], kps[[1, 1]]);
                let dx = ex1 - ex0;
                let dy = ey1 - ey0;
                angle = dy.atan2(dx) * 180.0 / std::f32::consts::PI;
            }
        }

        (cx, cy, side, angle)
    }

    /// 预处理: ROI 仿射变换 -> RGB -> 归一化 [0, 1] -> NCHW
    fn preprocess(
        &self,
        image: &DynamicImage,
        roi: &(f32, f32, f32, f32),
    ) -> Result<(ndarray::Array4<f32>, [[f32; 3]; 2])> {
        let (img_w, img_h) = image.dimensions();
        let rgb = image.to_rgb8();
        Self::preprocess_rgb(&rgb, img_w, img_h, roi, self.input_size)
    }

    /// preprocess 的 RGB 已转换版本 (batch 路径复用整图, 避免逐脸转换)
    fn preprocess_rgb(
        rgb: &image::RgbImage,
        img_w: u32,
        img_h: u32,
        roi: &(f32, f32, f32, f32),
        size: u32,
    ) -> Result<(ndarray::Array4<f32>, [[f32; 3]; 2])> {
        let (cx, cy, side, angle) = *roi;

        let scale = size as f32 / side;
        let rad = angle * std::f32::consts::PI / 180.0;
        let cos_a = rad.cos();
        let sin_a = rad.sin();

        let forward = [
            [
                cos_a * scale,
                -sin_a * scale,
                (-cx * cos_a + cy * sin_a) * scale + size as f32 / 2.0,
            ],
            [
                sin_a * scale,
                cos_a * scale,
                (-cx * sin_a - cy * cos_a) * scale + size as f32 / 2.0,
            ],
        ];

        let mut arr = ndarray::Array4::<f32>::zeros((1, 3, size as usize, size as usize));

        let inv = invert_affine(&forward);

        for y_out in 0..size {
            for x_out in 0..size {
                let sx = inv[0][0] * x_out as f32 + inv[0][1] * y_out as f32 + inv[0][2];
                let sy = inv[1][0] * x_out as f32 + inv[1][1] * y_out as f32 + inv[1][2];

                let sx_i = sx.round() as i32;
                let sy_i = sy.round() as i32;

                if sx_i >= 0 && sx_i < img_w as i32 && sy_i >= 0 && sy_i < img_h as i32 {
                    let pixel = rgb.get_pixel(sx_i as u32, sy_i as u32);
                    arr[[0, 0, y_out as usize, x_out as usize]] = pixel[0] as f32 / 255.0;
                    arr[[0, 1, y_out as usize, x_out as usize]] = pixel[1] as f32 / 255.0;
                    arr[[0, 2, y_out as usize, x_out as usize]] = pixel[2] as f32 / 255.0;
                }
            }
        }

        trace!(
            "FaceMesh: preprocessed {}x{} -> {}x{}, angle={:.1}°",
            img_w, img_h, size, size, angle
        );

        Ok((arr, inv))
    }

    /// 后处理: 将 crop 坐标反变换回原图坐标
    fn postprocess(
        landmarks: &[f32],
        logits: &[f32],
        inv_matrix: &[[f32; 3]; 2],
    ) -> Result<FaceMeshResult> {
        let n_points = 468;
        if landmarks.len() < n_points * 3 {
            anyhow::bail!(
                "FaceMesh output too short: {} floats, expected >= {}",
                landmarks.len(),
                n_points * 3
            );
        }
        let mut points_3d = Vec::with_capacity(n_points * 3);
        let mut points_2d = Vec::with_capacity(n_points * 2);

        for i in 0..n_points {
            let lx = landmarks[i * 3];
            let ly = landmarks[i * 3 + 1];
            let lz = landmarks[i * 3 + 2];

            let ox = inv_matrix[0][0] * lx + inv_matrix[0][1] * ly + inv_matrix[0][2];
            let oy = inv_matrix[1][0] * lx + inv_matrix[1][1] * ly + inv_matrix[1][2];

            points_3d.push(ox);
            points_3d.push(oy);
            points_3d.push(lz);
            points_2d.push(ox);
            points_2d.push(oy);
        }

        let score = if !logits.is_empty() {
            1.0 / (1.0 + (-logits[0]).exp())
        } else {
            1.0f32
        };

        let landmarks_3d = Array2::from_shape_vec((n_points, 3), points_3d)
            .map_err(|e| anyhow::anyhow!("Failed to create 3D landmarks array: {}", e))?;
        let landmarks_2d = Array2::from_shape_vec((n_points, 2), points_2d)
            .map_err(|e| anyhow::anyhow!("Failed to create 2D landmarks array: {}", e))?;

        Ok(FaceMeshResult {
            landmarks_3d,
            landmarks_2d,
            score,
        })
    }
}
