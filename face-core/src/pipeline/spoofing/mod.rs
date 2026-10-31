// ====================================================================
// 人脸活体检测 — MiniFASNet
// ====================================================================
//
// 裁剪方式: bbox 中心扩展裁剪,
//   scale = min((H-1)/box_h, (W-1)/box_w, self.scale), V1SE 用 4.0, V2 用 2.7;
//   裁剪后缩放到模型输入尺寸, 不做归一化 (原始 float32 [0, 255])。
// 后处理: softmax -> argmax,
//   标签索引 1 = 真人 (is_real), 置信度取该标签概率。单帧版。

use anyhow::Result;
use image::{DynamicImage, GenericImageView};
use log::{debug, info};
use ndarray::Array4;
use ort::value::TensorRef;

use crate::inference::InferenceSession;
use crate::types::{BBox, SpoofingResult};
use crate::utils::image_utils::resize_bilinear_cv2;

/// MiniFASNet V2 默认裁剪尺度 (V1SE 为 4.0)
pub const SCALE_V1SE: f32 = 4.0;
/// MiniFASNet V2 默认裁剪尺度
pub const SCALE_V2: f32 = 2.7;

/// MiniFASNet 活体检测器
pub struct MiniFASNet {
    session: InferenceSession,
    input_size: u32,
    scale: f32,
}

impl MiniFASNet {
    /// 从推理会话创建检测器 (V2 尺度 2.7)
    pub fn from_session(session: InferenceSession) -> Result<Self> {
        Self::from_session_with_scale(session, SCALE_V2)
    }

    /// 从推理会话创建检测器, 指定裁剪尺度 (V1SE: 4.0, V2: 2.7)
    pub fn from_session_with_scale(session: InferenceSession, scale: f32) -> Result<Self> {
        let input_size = 80u32; // MiniFASNet 输入 80x80 (input_size 从模型读取)
        info!(
            "MiniFASNet: input_size={}x{}, scale={}",
            input_size, input_size, scale
        );
        Ok(Self {
            session,
            input_size,
            scale,
        })
    }

    /// 端到端活体检测 (单帧)
    ///
    /// `bbox` 为人脸边界框 [x1, y1, x2, y2] (通常来自检测器)
    pub fn predict(&mut self, image: &DynamicImage, bbox: &BBox) -> Result<SpoofingResult> {
        let input_array = self.preprocess(image, bbox)?;

        let input = TensorRef::from_array_view(&input_array)?;
        let output_name = self.session.session().outputs()[0].name().to_string();
        let outputs = self.session.session_mut().run(ort::inputs![input])?;

        // 提取输出数据到 owned Vec (避免借用冲突)
        let (_shape, logits_ref) = outputs
            .get(output_name.as_str())
            .ok_or_else(|| anyhow::anyhow!("MiniFASNet model output '{}' not found", output_name))?
            .try_extract_tensor::<f32>()?;
        let logits: Vec<f32> = logits_ref.to_vec();
        drop(outputs);

        let result = parse_spoofing_logits(&logits);
        debug!(
            "MiniFASNet: is_real={}, confidence={:.4}",
            result.is_real, result.confidence
        );
        Ok(result)
    }

    /// 预处理: bbox 中心扩展裁剪 -> 缩放 -> CHW float32 (无归一化)
    fn preprocess(&self, image: &DynamicImage, bbox: &BBox) -> Result<Array4<f32>> {
        let (img_w, img_h) = image.dimensions();
        let size = self.input_size;

        // bbox 中心扩展裁剪: 先转 (x1,y1,w,h) 再计算扩展区域
        let x = bbox.x1 as i32;
        let y = bbox.y1 as i32;
        let box_w = (bbox.x2 - bbox.x1) as i32;
        let box_h = (bbox.y2 - bbox.y1) as i32;
        if box_w <= 0 || box_h <= 0 {
            anyhow::bail!("Invalid face bbox for anti-spoofing: {:?}", bbox);
        }

        let scale = ((img_h as f32 - 1.0) / box_h as f32)
            .min((img_w as f32 - 1.0) / box_w as f32)
            .min(self.scale);
        let new_w = box_w as f32 * scale;
        let new_h = box_h as f32 * scale;

        let center_x = x as f32 + box_w as f32 / 2.0;
        let center_y = y as f32 + box_h as f32 / 2.0;

        let x1 = (center_x - new_w / 2.0).max(0.0) as i32;
        let y1 = (center_y - new_h / 2.0).max(0.0) as i32;
        let x2 = (center_x + new_w / 2.0).min(img_w as f32 - 1.0).max(0.0) as i32;
        let y2 = (center_y + new_h / 2.0).min(img_h as f32 - 1.0).max(0.0) as i32;

        if x2 <= x1 || y2 <= y1 {
            anyhow::bail!("Degenerate crop region: ({},{})-({},{})", x1, y1, x2, y2);
        }

        // 裁剪区间 [y1:y2+1, x1:x2+1] 含端点
        let crop_w = (x2 - x1 + 1).max(1) as u32;
        let crop_h = (y2 - y1 + 1).max(1) as u32;
        let cropped = image.crop_imm(x1 as u32, y1 as u32, crop_w, crop_h);
        let resized = resize_bilinear_cv2(&cropped.to_rgb8(), size, size);

        // 模型输入为 BGR 且不交换通道; image crate 为 RGB, 需交换
        let mut arr = Array4::<f32>::zeros((1, 3, size as usize, size as usize));
        for (x, y, pixel) in resized.enumerate_pixels() {
            let c = &pixel.0;
            arr[[0, 0, y as usize, x as usize]] = c[2] as f32; // B
            arr[[0, 1, y as usize, x as usize]] = c[1] as f32; // G
            arr[[0, 2, y as usize, x as usize]] = c[0] as f32; // R
        }

        Ok(arr)
    }
}

/// 输出解析: softmax(logits) -> argmax, 索引 1 = 真人
pub fn parse_spoofing_logits(logits: &[f32]) -> SpoofingResult {
    if logits.is_empty() {
        return SpoofingResult {
            is_real: false,
            confidence: 0.0,
        };
    }
    let probs = softmax(logits);
    let label_idx = probs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    SpoofingResult {
        is_real: label_idx == 1,
        confidence: probs[label_idx],
    }
}

/// 数值稳定 softmax (减去最大值防溢出)
pub fn softmax(x: &[f32]) -> Vec<f32> {
    if x.is_empty() {
        return Vec::new();
    }
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = x.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_basic() {
        let probs = softmax(&[1.0, 2.0, 3.0]);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum={}", sum);
        assert!(probs[0] < probs[1] && probs[1] < probs[2]);
    }

    #[test]
    fn test_softmax_stability() {
        // 极大 logits 不应溢出
        let probs = softmax(&[1000.0, 1001.0]);
        assert!(probs.iter().all(|p| p.is_finite()));
        assert!((probs[0] + probs[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_parse_spoofing_real() {
        // 索引 1 得分最高 -> is_real=true
        let result = parse_spoofing_logits(&[-2.0, 3.0]);
        assert!(result.is_real);
        assert!(
            (result.confidence - 0.9933).abs() < 1e-2,
            "conf={}",
            result.confidence
        );
    }

    #[test]
    fn test_parse_spoofing_fake() {
        // 索引 0 得分最高 -> is_real=false
        let result = parse_spoofing_logits(&[5.0, -1.0]);
        assert!(!result.is_real);
        assert!(result.confidence > 0.99);
    }

    #[test]
    fn test_parse_spoofing_empty() {
        // 空输出: is_real=false, confidence=0
        let result = parse_spoofing_logits(&[]);
        assert!(!result.is_real);
        assert_eq!(result.confidence, 0.0);
    }

    #[test]
    fn test_parse_spoofing_confidence_matches_softmax() {
        let logits = [0.3, -0.7];
        let result = parse_spoofing_logits(&logits);
        let probs = softmax(&logits);
        assert!((result.confidence - probs[0]).abs() < 1e-6);
        assert!(!result.is_real);
    }
}
