// ====================================================================
// BlazeFace 人脸检测器 — 轻量级移动端人脸检测实现
// ====================================================================
//
// 模型: BlazeFace 短距离检测器
//       - 输入: 128x128 RGB, 归一化到 [-1, 1]
//       - 输出: 896 个锚点的回归+分类结果, 带 6 个关键点
//       - 模型大小: ~460KB
//       - 特点: 专为移动端 GPU 设计, 使用加权 NMS

use anyhow::Result;
use image::{DynamicImage, GenericImageView};
use log::{debug, info, trace};
use ndarray::Array2;
use ort::value::TensorRef;

use crate::inference::InferenceSession;
use crate::types::{BBox, DetectedFace};
use crate::utils::common::{clamp_f32, compute_iou, sigmoid};

/// BlazeFace 检测器 — 面向移动端的轻量级人脸检测
pub struct BlazeFaceDetector {
    session: InferenceSession,
    confidence_threshold: f32,
    nms_threshold: f32,
    anchors: Array2<f32>,
}

/// BlazeFace 锚点生成配置 (4层 SSD, 步长 [8, 16, 16, 16])
const ANCHOR_STRIDES: [usize; 4] = [8, 16, 16, 16];

impl BlazeFaceDetector {
    /// 从内嵌模型创建 BlazeFace 检测器
    pub fn from_session(
        session: InferenceSession,
        confidence_threshold: f32,
        nms_threshold: f32,
    ) -> Result<Self> {
        info!(
            "BlazeFace: confidence_threshold={}, nms_threshold={}",
            confidence_threshold, nms_threshold
        );
        let anchors = Self::generate_anchors();
        Ok(Self {
            session,
            confidence_threshold,
            nms_threshold,
            anchors,
        })
    }

    /// 生成 BlazeFace 的 896 个 SSD 锚点
    ///
    /// 同 stride 的层合并, 每格重复 repeats = 2 * (同 stride 层数)
    /// 次 → 16x16x2 + 8x8x6 = 896
    fn generate_anchors() -> Array2<f32> {
        let input_size: usize = 128;

        let mut anchors: Vec<(f32, f32)> = Vec::with_capacity(896);
        let mut idx = 0;
        while idx < ANCHOR_STRIDES.len() {
            // 找出共享同一 stride 的连续层组
            let mut last = idx;
            while last < ANCHOR_STRIDES.len() && ANCHOR_STRIDES[last] == ANCHOR_STRIDES[idx] {
                last += 1;
            }
            let repeats = 2 * (last - idx);
            let cells = input_size / ANCHOR_STRIDES[idx];
            for y in 0..cells {
                for x in 0..cells {
                    for _ in 0..repeats {
                        anchors.push((
                            (x as f32 + 0.5) / cells as f32,
                            (y as f32 + 0.5) / cells as f32,
                        ));
                    }
                }
            }
            idx = last;
        }

        let n = anchors.len();
        debug!("BlazeFace: generated {} anchors", n);
        let mut arr = Array2::<f32>::zeros((n, 2));
        for (i, &(cx, cy)) in anchors.iter().enumerate() {
            arr[[i, 0]] = cx;
            arr[[i, 1]] = cy;
        }
        arr
    }

    /// 检测图像中的所有人脸
    pub fn detect(&mut self, image: &DynamicImage) -> Result<Vec<DetectedFace>> {
        let (img_w, img_h) = image.dimensions();
        debug!("BlazeFace: detecting faces in {}x{} image", img_w, img_h);

        // 1. 预处理: letterbox 缩放到 128x128
        let (input_array, (scale, pad_x, pad_y)) = self.preprocess(image)?;

        // 2. ONNX 推理
        let input = TensorRef::from_array_view(&input_array)?;
        let outputs = self.session.session_mut().run(ort::inputs![input])?;

        // 提取输出数据到 owned Vec (避免借用冲突; 输出名缺失返回 Err 而非 panic)
        let regressors_ref = outputs
            .get("regressors")
            .ok_or_else(|| anyhow::anyhow!("BlazeFace model output 'regressors' not found"))?
            .try_extract_tensor::<f32>()?;
        let logits_ref = outputs
            .get("scores")
            .ok_or_else(|| anyhow::anyhow!("BlazeFace model output 'scores' not found"))?
            .try_extract_tensor::<f32>()?;
        let (_reg_shape, regressors_ref) = regressors_ref;
        let (_log_shape, logits_ref) = logits_ref;
        let regressors: Vec<f32> = regressors_ref.to_vec();
        let logits: Vec<f32> = logits_ref.to_vec();
        drop(outputs);

        debug!(
            "BlazeFace inference: regressors len={}, logits len={}",
            regressors.len(),
            logits.len()
        );

        // 3. 后处理: 解码锚点、加权NMS
        let faces = self.postprocess(&regressors, &logits, scale, pad_x, pad_y, img_w, img_h);

        info!(
            "BlazeFace: detected {} face(s) in {}x{} image",
            faces.len(),
            img_w,
            img_h
        );

        Ok(faces)
    }

    /// 预处理: BGR -> RGB -> letterbox 128x128 -> 归一化到 [-1, 1] -> NCHW
    fn preprocess(&self, image: &DynamicImage) -> Result<(ndarray::Array4<f32>, (f32, f32, f32))> {
        let (img_w, img_h) = image.dimensions();
        let rgb = image.to_rgb8();

        let canvas_w: u32 = 128;
        let canvas_h: u32 = 128;
        let scale = canvas_w as f32 / img_w.max(img_h) as f32;
        let scaled_w = (img_w as f32 * scale).round() as u32;
        let scaled_h = (img_h as f32 * scale).round() as u32;
        let pad_x = (canvas_w - scaled_w) as f32 / 2.0;
        let pad_y = (canvas_h - scaled_h) as f32 / 2.0;

        let resized = image::imageops::resize(
            &rgb,
            scaled_w,
            scaled_h,
            image::imageops::FilterType::Lanczos3,
        );

        // 将缩放后的图像放入 128x128 canvas
        let mut canvas = vec![0u8; (canvas_w * canvas_h * 3) as usize];
        for y in 0..scaled_h {
            for x in 0..scaled_w {
                let pixel = resized.get_pixel(x, y);
                let dst_idx = (((y + pad_y as u32) * canvas_w + x + pad_x as u32) * 3) as usize;
                canvas[dst_idx] = pixel[0];
                canvas[dst_idx + 1] = pixel[1];
                canvas[dst_idx + 2] = pixel[2];
            }
        }

        let mut arr = ndarray::Array4::<f32>::zeros((1, 3, 128, 128));
        for y in 0..128usize {
            for x in 0..128usize {
                let idx = (y * 128 + x) * 3;
                arr[[0, 0, y, x]] = (canvas[idx] as f32 - 127.5) / 127.5;
                arr[[0, 1, y, x]] = (canvas[idx + 1] as f32 - 127.5) / 127.5;
                arr[[0, 2, y, x]] = (canvas[idx + 2] as f32 - 127.5) / 127.5;
            }
        }

        trace!(
            "BlazeFace: preprocessed -> (1,3,128,128), scale={:.4}, pad=({:.1},{:.1})",
            scale, pad_x, pad_y
        );

        Ok((arr, (scale, pad_x, pad_y)))
    }

    /// 后处理: 解码回归值、应用加权NMS、恢复原始坐标
    fn postprocess(
        &self,
        regressors: &[f32],
        logits: &[f32],
        scale: f32,
        pad_x: f32,
        pad_y: f32,
        img_w: u32,
        img_h: u32,
    ) -> Vec<DetectedFace> {
        let n_anchors = self.anchors.shape()[0];

        let mut detections: Vec<(f32, f32, f32, f32, f32, [(f32, f32); 6])> = Vec::new();

        for i in 0..n_anchors {
            let raw_logit = logits[i].clamp(-100.0, 100.0);
            let score = sigmoid(raw_logit);

            if score < self.confidence_threshold {
                continue;
            }

            let anchor_cx = self.anchors[[i, 0]];
            let anchor_cy = self.anchors[[i, 1]];
            let size = 128.0f32;

            // 回归布局: [cx, cy, w, h, kp0x, kp0y, ..., kp5y] 共 16 维
            let cx = regressors[i * 16] / size + anchor_cx;
            let cy = regressors[i * 16 + 1] / size + anchor_cy;
            let w = regressors[i * 16 + 2] / size;
            let h = regressors[i * 16 + 3] / size;

            let mut keypoints = [(0.0f32, 0.0f32); 6];
            for (k, kp) in keypoints.iter_mut().enumerate() {
                let kp_x = regressors[i * 16 + 4 + k * 2] / size + anchor_cx;
                let kp_y = regressors[i * 16 + 5 + k * 2] / size + anchor_cy;
                *kp = (kp_x, kp_y);
            }

            detections.push((score, cx, cy, w, h, keypoints));
        }

        let merged = self.weighted_nms(&detections);

        let mut faces = Vec::new();
        for (score, cx, cy, w, h, keypoints) in merged {
            let canvas_x1 = (cx - w / 2.0) * 128.0;
            let canvas_y1 = (cy - h / 2.0) * 128.0;
            let canvas_x2 = (cx + w / 2.0) * 128.0;
            let canvas_y2 = (cy + h / 2.0) * 128.0;

            let img_x1 = clamp_f32((canvas_x1 - pad_x) / scale, 0.0, img_w as f32);
            let img_y1 = clamp_f32((canvas_y1 - pad_y) / scale, 0.0, img_h as f32);
            let img_x2 = clamp_f32((canvas_x2 - pad_x) / scale, 0.0, img_w as f32);
            let img_y2 = clamp_f32((canvas_y2 - pad_y) / scale, 0.0, img_h as f32);

            let img_kps: Vec<(f32, f32)> = keypoints
                .iter()
                .map(|(kx, ky)| {
                    let px = clamp_f32((kx * 128.0 - pad_x) / scale, 0.0, img_w as f32);
                    let py = clamp_f32((ky * 128.0 - pad_y) / scale, 0.0, img_h as f32);
                    (px, py)
                })
                .collect();

            let kp_array = Array2::from_shape_vec(
                (6, 2),
                img_kps.iter().flat_map(|(x, y)| vec![*x, *y]).collect(),
            )
            .ok();

            faces.push(DetectedFace {
                bbox: BBox::new(img_x1, img_y1, img_x2, img_y2),
                confidence: score,
                landmarks: kp_array,
                mesh_3d: None,
                mesh_score: None,
                track_id: None,
                embedding: None,
                gender: None,
                age: None,
                age_group: None,
                race: None,
                emotion: None,
                emotion_confidence: None,
                face_state: None,
                quality: None,
                gaze: None,
                head_pose: None,
                spoofing: None,
            });
        }

        faces
    }

    /// 加权NMS: 重叠框加权融合而非丢弃
    fn weighted_nms(
        &self,
        detections: &[(f32, f32, f32, f32, f32, [(f32, f32); 6])],
    ) -> Vec<(f32, f32, f32, f32, f32, [(f32, f32); 6])> {
        let n = detections.len();
        if n == 0 {
            return Vec::new();
        }

        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            detections[b]
                .0
                .partial_cmp(&detections[a].0)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut remaining: Vec<bool> = vec![true; n];
        let mut result = Vec::new();

        for &i in &indices {
            if !remaining[i] {
                continue;
            }

            let (_score_i, cx_i, cy_i, w_i, h_i, _kps_i) = &detections[i];
            let box_i = (
                cx_i - w_i / 2.0,
                cy_i - h_i / 2.0,
                cx_i + w_i / 2.0,
                cy_i + h_i / 2.0,
            );

            let mut cluster: Vec<usize> = vec![i];
            for &j in &indices {
                if i == j || !remaining[j] {
                    continue;
                }
                let (_, cx_j, cy_j, w_j, h_j, _) = &detections[j];
                let box_j = (
                    cx_j - w_j / 2.0,
                    cy_j - h_j / 2.0,
                    cx_j + w_j / 2.0,
                    cy_j + h_j / 2.0,
                );
                if compute_iou(
                    (box_i.0, box_i.1, box_i.2, box_i.3),
                    (box_j.0, box_j.1, box_j.2, box_j.3),
                ) > self.nms_threshold
                {
                    cluster.push(j);
                    remaining[j] = false;
                }
            }
            remaining[i] = false;

            let total_score: f32 = cluster.iter().map(|&idx| detections[idx].0).sum();
            let mut avg_cx = 0.0f32;
            let mut avg_cy = 0.0f32;
            let mut avg_w = 0.0f32;
            let mut avg_h = 0.0f32;
            let mut avg_kps = [(0.0f32, 0.0f32); 6];

            for &idx in &cluster {
                let weight = detections[idx].0 / total_score;
                avg_cx += detections[idx].1 * weight;
                avg_cy += detections[idx].2 * weight;
                avg_w += detections[idx].3 * weight;
                avg_h += detections[idx].4 * weight;
                for k in 0..6 {
                    avg_kps[k].0 += detections[idx].5[k].0 * weight;
                    avg_kps[k].1 += detections[idx].5[k].1 * weight;
                }
            }

            let top_score = detections[i].0;
            result.push((top_score, avg_cx, avg_cy, avg_w, avg_h, avg_kps));
        }

        result.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        result
    }
}
