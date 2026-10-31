// ====================================================================
// 视线估计 — MobileNetV2 分类头 (Gaze360 分箱解码)
// ====================================================================
//
// 视线估计统一实现 (MobileGaze)
//
// 公共流程:
//   裁脸 → resize 224x224 → ImageNet 归一化 → ONNX 推理
//   → 两个 90-bin logits (yaw, pitch) → softmax → 期望 bin → 度转弧度
//
// 模型来源:

use anyhow::Result;
use image::{DynamicImage, GenericImageView};
use log::{info, trace};
use ndarray::Array4;
use ort::value::TensorRef;
use rayon::prelude::*;

use crate::inference::InferenceSession;
use crate::types::{BBox, DetectedFace, GazeResult};
use crate::utils::image_utils::resize_bilinear_cv2;

/// ImageNet 均值 (RGB)
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
/// ImageNet 标准差 (RGB)
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Gaze360 分箱参数
#[cfg(test)]
const GAZE_BINS: usize = 90;
const GAZE_BINWIDTH: f32 = 4.0;
const GAZE_ANGLE_OFFSET: f32 = 180.0;

/// 视线估计模型 (MobileNetV2)
///
/// - 输入: bbox 裁脸 → 224x224 RGB → ImageNet 归一化
/// - 输出: yaw / pitch 两个 90-bin logits → softmax 期望值 → 弧度
pub struct GazeNet {
    session: InferenceSession,
    input_size: u32,
}

impl GazeNet {
    /// 创建 GazeNet
    ///
    /// 输入尺寸从模型 input shape 动态读取 (ONNX 元数据;
    /// mobilenetv2_gaze 权重实际为 448x448)
    pub fn from_session(session: InferenceSession) -> Result<Self> {
        let input_size = session
            .session()
            .inputs()
            .first()
            .and_then(|input| input.dtype().tensor_shape())
            .and_then(|shape| shape.get(3).copied())
            .filter(|&d| d > 0)
            .unwrap_or(224) as u32;
        info!("GazeNet: input_size={}x{}", input_size, input_size);
        Ok(Self {
            session,
            input_size,
        })
    }

    /// 预测视线方向
    pub fn predict(&mut self, image: &DynamicImage, face: &DetectedFace) -> Result<GazeResult> {
        let input_array = self.preprocess(image, &face.bbox)?;

        let input = TensorRef::from_array_view(&input_array)?;

        // 模型有两个输出 (yaw, pitch); 按声明顺序提取 (先取名字再推理, 避免借用冲突)
        let out_names: Vec<String> = self
            .session
            .session()
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        if out_names.len() < 2 {
            anyhow::bail!(
                "Gaze model expected 2 outputs (yaw, pitch), got {}",
                out_names.len()
            );
        }

        let outputs = self.session.session_mut().run(ort::inputs![input])?;
        let yaw_ref = outputs
            .get(out_names[0].as_str())
            .ok_or_else(|| anyhow::anyhow!("Gaze model output '{}' not found", out_names[0]))?
            .try_extract_tensor::<f32>()?;
        let pitch_ref = outputs
            .get(out_names[1].as_str())
            .ok_or_else(|| anyhow::anyhow!("Gaze model output '{}' not found", out_names[1]))?
            .try_extract_tensor::<f32>()?;
        let (_s0, yaw_ref) = yaw_ref;
        let (_s1, pitch_ref) = pitch_ref;
        let yaw_logits: Vec<f32> = yaw_ref.to_vec();
        let pitch_logits: Vec<f32> = pitch_ref.to_vec();
        drop(outputs);

        let result = parse_gaze_output(&yaw_logits, &pitch_logits)?;

        trace!(
            "GazeNet: pitch={:.3} yaw={:.3} rad",
            result.pitch, result.yaw
        );
        Ok(result)
    }

    /// 批量预测视线方向: 逐脸预处理后拼 (N,3,H,W) 单次推理
    ///
    /// 返回按输入顺序对齐的结果; 预处理失败的脸为 None
    pub fn predict_batch(
        &mut self,
        image: &DynamicImage,
        faces: &[&DetectedFace],
    ) -> Result<Vec<Option<GazeResult>>> {
        if faces.is_empty() {
            return Ok(Vec::new());
        }
        let size = self.input_size as usize;
        // 预处理互相独立, rayon 并行裁剪
        let prepped: Vec<Result<Array4<f32>>> = faces
            .par_iter()
            .map(|face| self.preprocess(image, &face.bbox))
            .collect();
        let mut inputs = Vec::with_capacity(faces.len());
        let mut valid = Vec::with_capacity(faces.len());
        for (i, r) in prepped.into_iter().enumerate() {
            match r {
                Ok(blob) => {
                    inputs.push(blob);
                    valid.push(i);
                }
                Err(e) => {
                    log::debug!("GazeNet: face {} preprocessing failed: {}", i, e);
                }
            }
        }
        let mut results: Vec<Option<GazeResult>> = vec![None; faces.len()];
        if inputs.is_empty() {
            return Ok(results);
        }

        let n = inputs.len();
        let mut batch = ndarray::Array4::<f32>::zeros((n, 3, size, size));
        for (b, blob) in inputs.iter().enumerate() {
            batch
                .slice_mut(ndarray::s![b..b + 1, .., .., ..])
                .assign(blob);
        }

        let tensor = TensorRef::from_array_view(&batch)?;
        let out_names: Vec<String> = self
            .session
            .session()
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        if out_names.len() < 2 {
            anyhow::bail!(
                "Gaze model expected 2 outputs (yaw, pitch), got {}",
                out_names.len()
            );
        }
        let outputs = self.session.session_mut().run(ort::inputs![tensor])?;
        let yaw_ref = outputs
            .get(out_names[0].as_str())
            .ok_or_else(|| anyhow::anyhow!("Gaze model output '{}' not found", out_names[0]))?
            .try_extract_tensor::<f32>()?;
        let pitch_ref = outputs
            .get(out_names[1].as_str())
            .ok_or_else(|| anyhow::anyhow!("Gaze model output '{}' not found", out_names[1]))?
            .try_extract_tensor::<f32>()?;
        let (_s0, yaw_ref) = yaw_ref;
        let (_s1, pitch_ref) = pitch_ref;
        let yaw_flat: Vec<f32> = yaw_ref.to_vec();
        let pitch_flat: Vec<f32> = pitch_ref.to_vec();
        drop(outputs);

        let dim = yaw_flat.len() / n;
        for (b, &i) in valid.iter().enumerate() {
            let yaw = &yaw_flat[b * dim..(b + 1) * dim];
            let pitch = &pitch_flat[b * dim..(b + 1) * dim];
            if let Ok(r) = parse_gaze_output(yaw, pitch) {
                results[i] = Some(r);
            }
        }
        Ok(results)
    }

    /// 预处理: bbox 裁脸 → resize → ImageNet 归一化 → NCHW
    fn preprocess(&self, image: &DynamicImage, bbox: &BBox) -> Result<Array4<f32>> {
        let (img_w, img_h) = image.dimensions();
        let size = self.input_size;

        // 像素坐标截断 + 像素区间裁剪 (排除右/下端点)
        let x1 = (bbox.x1 as i32).max(0);
        let y1 = (bbox.y1 as i32).max(0);
        let x2 = (bbox.x2 as i32).min(img_w as i32);
        let y2 = (bbox.y2 as i32).min(img_h as i32);

        if x2 <= x1 || y2 <= y1 {
            anyhow::bail!(
                "GazeNet: bbox ({},{})-({},{}) does not overlap the {}x{} image",
                bbox.x1,
                bbox.y1,
                bbox.x2,
                bbox.y2,
                img_w,
                img_h
            );
        }

        let crop = image.crop_imm(x1 as u32, y1 as u32, (x2 - x1) as u32, (y2 - y1) as u32);
        let resized = resize_bilinear_cv2(&crop.to_rgb8(), size, size);
        let rgb = resized;

        let mut arr = Array4::<f32>::zeros((1, 3, size as usize, size as usize));
        for y in 0..size {
            for x in 0..size {
                let p = rgb.get_pixel(x, y);
                for c in 0..3 {
                    let v = p[c] as f32 / 255.0;
                    arr[[0, c, y as usize, x as usize]] = (v - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
                }
            }
        }

        Ok(arr)
    }
}

/// softmax: exp(x - max) / sum, 数值稳定
fn softmax(values: &[f32]) -> Vec<f32> {
    if values.is_empty() {
        return Vec::new();
    }
    let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = values.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        return vec![1.0 / values.len() as f32; values.len()];
    }
    exps.iter().map(|e| e / sum).collect()
}

/// 解析 Gaze 模型输出
///
/// 两个 90-bin logits → softmax → 期望 bin index × binwidth - offset → 弧度
pub fn parse_gaze_output(yaw_logits: &[f32], pitch_logits: &[f32]) -> Result<GazeResult> {
    if yaw_logits.is_empty() || pitch_logits.is_empty() {
        anyhow::bail!(
            "Gaze logits are empty (yaw={}, pitch={})",
            yaw_logits.len(),
            pitch_logits.len()
        );
    }

    let soft_argmax = |logits: &[f32]| -> f32 {
        let probs = softmax(logits);
        let idx_sum: f32 = probs.iter().enumerate().map(|(i, &p)| p * i as f32).sum();
        idx_sum * GAZE_BINWIDTH - GAZE_ANGLE_OFFSET
    };

    let yaw_deg = soft_argmax(yaw_logits);
    let pitch_deg = soft_argmax(pitch_logits);

    Ok(GazeResult {
        pitch: pitch_deg.to_radians(),
        yaw: yaw_deg.to_radians(),
    })
}

// --------------------------------------------------------------------
// 单元测试
// --------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_sums_to_one() {
        let p = softmax(&[1.0, 2.0, 3.0]);
        assert_eq!(p.len(), 3);
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // 单调性: logit 越大概率越大
        assert!(p[2] > p[1] && p[1] > p[0]);
    }

    #[test]
    fn test_softmax_extreme_values() {
        let p = softmax(&[1000.0, 0.0, -1000.0]);
        assert!((p[0] - 1.0).abs() < 1e-6);
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_parse_gaze_output_one_hot() {
        // 单一 bin 概率 1 → 角度 = idx * 4 - 180 度
        let mut yaw_logits = vec![0.0f32; GAZE_BINS];
        let mut pitch_logits = vec![0.0f32; GAZE_BINS];
        // logit=50: softmax 接近严格 one-hot (logit=10 在边缘 bin 89 仍有 ~0.7° 偏差)
        yaw_logits[45] = 50.0; // 45 * 4 - 180 = 0°
        pitch_logits[90 - 1] = 50.0; // 89 * 4 - 180 = 176°

        let r = parse_gaze_output(&yaw_logits, &pitch_logits).unwrap();
        // softmax 非严格 one-hot: 允许 1e-3 弧度级误差
        assert!(r.yaw.abs() < 1e-3, "yaw should be ~0, got {}", r.yaw);
        let expect_pitch = 176.0f32.to_radians();
        assert!(
            (r.pitch - expect_pitch).abs() < 1e-3,
            "pitch should be ~176°, got {}",
            r.pitch
        );
    }

    #[test]
    fn test_parse_gaze_output_uniform() {
        // 均匀 logits → 期望 bin = (90-1)/2 = 44.5 → -2° → 弧度
        let logits = vec![0.0f32; GAZE_BINS];
        let r = parse_gaze_output(&logits, &logits).unwrap();
        let expect = (44.5f32 * GAZE_BINWIDTH - GAZE_ANGLE_OFFSET).to_radians();
        assert!((r.yaw - expect).abs() < 1e-5);
        assert!((r.pitch - expect).abs() < 1e-5);
    }

    #[test]
    fn test_parse_gaze_output_empty_errors() {
        assert!(parse_gaze_output(&[], &[0.0]).is_err());
        assert!(parse_gaze_output(&[0.0], &[]).is_err());
    }
}
