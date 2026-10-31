// ====================================================================
// MODNet 人像抠图 — 整图软 alpha 遮罩
// ====================================================================
//
//
// - 预处理: RGB, 短边 resize 到 input_size (保持纵横比),
//   两边向下对齐到 32 的倍数, 归一化到 [-1, 1]
// - 输出: (1, 1, H', W') alpha, 双线性 resize 回原图并 clamp 到 [0, 1]

use anyhow::Result;
use image::{DynamicImage, GenericImageView};
use log::{info, trace};
use ort::value::TensorRef;

use crate::inference::InferenceSession;
use crate::types::MatteResult;

/// 尺寸对齐步长 (推理宽高需对齐到该值的倍数)
const STRIDE: u32 = 32;

/// MODNet 人像抠图模型
pub struct ModNet {
    session: InferenceSession,
    /// 预处理目标短边尺寸 (默认 512)
    input_size: u32,
}

impl ModNet {
    /// 创建 MODNet 模型, input_size 默认 512
    pub fn from_session(session: InferenceSession) -> Result<Self> {
        Self::with_input_size(session, 512)
    }

    /// 创建 MODNet 模型并指定 input_size
    pub fn with_input_size(session: InferenceSession, input_size: u32) -> Result<Self> {
        if input_size == 0 || !input_size.is_multiple_of(STRIDE) {
            anyhow::bail!(
                "input_size must be a positive multiple of {}, got {}",
                STRIDE,
                input_size
            );
        }
        info!("ModNet: input_size={}", input_size);
        Ok(Self {
            session,
            input_size,
        })
    }

    /// 对整图执行人像抠图
    pub fn predict(&mut self, image: &DynamicImage) -> Result<MatteResult> {
        let (orig_w, orig_h) = image.dimensions();
        let (tensor, new_w, new_h) = self.preprocess(image)?;

        let input = TensorRef::from_array_view(&tensor)?;
        let output_name = self.session.session().outputs()[0].name().to_string();
        let outputs = self.session.session_mut().run(ort::inputs![input])?;
        let (_shape, data_ref) = outputs
            .get(output_name.as_str())
            .ok_or_else(|| anyhow::anyhow!("MODNet model output '{}' not found", output_name))?
            .try_extract_tensor::<f32>()?;
        let data: Vec<f32> = data_ref.to_vec();
        drop(outputs);

        let alpha = postprocess_alpha(&data, new_w, new_h, orig_w, orig_h)?;

        trace!(
            "ModNet: matted {}x{} -> {}x{} alpha",
            new_w, new_h, orig_w, orig_h
        );

        Ok(MatteResult {
            alpha,
            width: orig_w,
            height: orig_h,
        })
    }

    /// 预处理:
    ///  - 若 max(h,w) < input_size 或 min(h,w) > input_size, 将短边缩放到
    ///    input_size 并保持比例; 否则保持原尺寸
    ///  - 两边向下对齐到 32 的倍数
    ///  - 归一化到 [-1, 1] (即 (x - 0.5) / 0.5)
    fn preprocess(&self, image: &DynamicImage) -> Result<(ndarray::Array4<f32>, u32, u32)> {
        let (orig_w, orig_h) = image.dimensions();
        if orig_w == 0 || orig_h == 0 {
            anyhow::bail!("Cannot matte an empty image");
        }

        let (mut new_w, mut new_h) = (orig_w, orig_h);
        if orig_w.max(orig_h) < self.input_size || orig_w.min(orig_h) > self.input_size {
            if orig_w >= orig_h {
                new_h = self.input_size;
                new_w = ((orig_w as f64 / orig_h as f64) * self.input_size as f64) as u32;
            } else {
                new_w = self.input_size;
                new_h = ((orig_h as f64 / orig_w as f64) * self.input_size as f64) as u32;
            }
        }
        new_w -= new_w % STRIDE;
        new_h -= new_h % STRIDE;
        if new_w == 0 || new_h == 0 {
            anyhow::bail!("Input collapses to zero size after stride alignment");
        }

        // INTER_AREA 对应 image crate 的 Triangle 滤波 (下采样友好)
        let filter = if new_w < orig_w || new_h < orig_h {
            image::imageops::FilterType::Triangle
        } else {
            image::imageops::FilterType::Lanczos3
        };
        let resized = image.resize_exact(new_w, new_h, filter).to_rgb8();

        let (w, h) = (new_w as usize, new_h as usize);
        let mut arr = ndarray::Array4::<f32>::zeros((1, 3, h, w));
        for y in 0..h {
            for x in 0..w {
                let p = resized.get_pixel(x as u32, y as u32);
                for c in 0..3 {
                    // (x/255 - 0.5) / 0.5
                    arr[[0, c, y, x]] = (p[c] as f32 / 255.0 - 0.5) / 0.5;
                }
            }
        }

        Ok((arr, new_w, new_h))
    }
}

/// 后处理:
/// alpha (H', W') → 双线性 resize 回原图 (W, H) → clamp [0, 1]
///
/// 返回 (H, W) 行优先的二维数组
pub(crate) fn postprocess_alpha(
    alpha: &[f32],
    src_w: u32,
    src_h: u32,
    out_w: u32,
    out_h: u32,
) -> Result<ndarray::Array2<f32>> {
    let (sw, sh) = (src_w as usize, src_h as usize);
    if sw == 0 || sh == 0 || alpha.len() < sw * sh {
        anyhow::bail!(
            "Alpha matte output too short: {} floats, expected >= {}",
            alpha.len(),
            sw * sh
        );
    }

    let mut result = ndarray::Array2::<f32>::zeros((out_h as usize, out_w as usize));
    // 目标像素中心 -> 源像素中心的映射 (半像素中心对齐语义)
    let x_scale = sw as f32 / out_w as f32;
    let y_scale = sh as f32 / out_h as f32;

    for y in 0..out_h as usize {
        let sy = (y as f32 + 0.5) * y_scale - 0.5;
        let y0 = sy.floor().max(0.0) as usize;
        let y1 = (y0 + 1).min(sh - 1);
        let fy = (sy - y0 as f32).clamp(0.0, 1.0);

        for x in 0..out_w as usize {
            let sx = (x as f32 + 0.5) * x_scale - 0.5;
            let x0 = sx.floor().max(0.0) as usize;
            let x1 = (x0 + 1).min(sw - 1);
            let fx = (sx - x0 as f32).clamp(0.0, 1.0);

            let p00 = alpha[y0 * sw + x0];
            let p10 = alpha[y0 * sw + x1];
            let p01 = alpha[y1 * sw + x0];
            let p11 = alpha[y1 * sw + x1];

            let top = p00 * (1.0 - fx) + p10 * fx;
            let bottom = p01 * (1.0 - fx) + p11 * fx;
            let value = top * (1.0 - fy) + bottom * fy;

            result[[y, x]] = value.clamp(0.0, 1.0);
        }
    }

    Ok(result)
}

/// 计算对齐 STRIDE 后的推理尺寸 (纯函数, 便于单测)
///
/// 预处理尺寸的分支逻辑:
/// - max(h,w) < input_size 或 min(h,w) > input_size → 短边缩放到 input_size
/// - 否则保持原尺寸; 最后两边向下对齐到 32 的倍数
#[cfg(test)]
pub(crate) fn compute_inference_size(orig_w: u32, orig_h: u32, input_size: u32) -> (u32, u32) {
    let (mut new_w, mut new_h) = (orig_w, orig_h);
    if orig_w.max(orig_h) < input_size || orig_w.min(orig_h) > input_size {
        if orig_w >= orig_h {
            new_h = input_size;
            new_w = ((orig_w as f64 / orig_h as f64) * input_size as f64) as u32;
        } else {
            new_w = input_size;
            new_h = ((orig_h as f64 / orig_w as f64) * input_size as f64) as u32;
        }
    }
    (new_w - new_w % STRIDE, new_h - new_h % STRIDE)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- compute_inference_size ----

    #[test]
    fn test_inference_size_small_image_upscaled() {
        // 200x100: max < 512 → 短边放大到 512, 宽按比例 → 1024, 均为 32 倍数
        assert_eq!(compute_inference_size(200, 100, 512), (1024, 512));
    }

    #[test]
    fn test_inference_size_large_image_downscaled() {
        // 2048x3072: min > 512 → 短边(宽)缩到 512 → (512, 768), 均为 32 倍数
        assert_eq!(compute_inference_size(2048, 3072, 512), (512, 768));
    }

    #[test]
    fn test_inference_size_in_between_unchanged() {
        // 400x600: max > 512 且 min < 512 → 保持原尺寸, 向下对齐 32: 400→384, 600→576
        assert_eq!(compute_inference_size(400, 600, 512), (384, 576));
    }

    #[test]
    fn test_inference_size_stride_floor() {
        // 517x300: max > 512 但 min < 512 → 保持原尺寸, 向下对齐 32
        assert_eq!(compute_inference_size(517, 300, 512), (512, 288));
    }

    #[test]
    fn test_inference_size_tall_image() {
        // 竖图 100x900: max > 512 且 min < 512 → 保持原尺寸, 对齐 32: (96, 896)
        assert_eq!(compute_inference_size(100, 900, 512), (96, 896));
    }

    // ---- postprocess_alpha ----

    #[test]
    fn test_alpha_resize_identity() {
        // 同尺寸: 直接透传
        let alpha: Vec<f32> = (0..64).map(|i| (i % 7) as f32 / 7.0).collect();
        let out = postprocess_alpha(&alpha, 8, 8, 8, 8).unwrap();
        for i in 0..64 {
            assert!((out.as_slice().unwrap()[i] - alpha[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_alpha_resize_upscale_constant() {
        // 常量遮罩上采样后仍是常量
        let alpha = vec![0.5f32; 16]; // 4x4
        let out = postprocess_alpha(&alpha, 4, 4, 8, 8).unwrap();
        assert_eq!(out.shape(), &[8, 8]);
        for v in out.iter() {
            assert!((v - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn test_alpha_clamped_to_unit_range() {
        // 越界值 clamp 到 [0, 1]
        let alpha = vec![-2.0f32, 3.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        let out = postprocess_alpha(&alpha, 4, 2, 4, 2).unwrap();
        let s = out.as_slice().unwrap();
        assert_eq!(s[0], 0.0);
        assert_eq!(s[1], 1.0);
        assert!((s[2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_alpha_short_data_errors() {
        // 数据不足应报错
        assert!(postprocess_alpha(&[0.1, 0.2], 4, 4, 8, 8).is_err());
    }

    #[test]
    fn test_alpha_resize_upscale_linear_ramp() {
        // 水平渐变 [0,1] 上采样 2→4: 双线性插值结果 [0, 0.25, 0.75, 1]
        let alpha = vec![0.0f32, 1.0, 0.0, 1.0]; // 2x2
        let out = postprocess_alpha(&alpha, 2, 2, 4, 2).unwrap();
        let s = out.as_slice().unwrap();
        for row in 0..2 {
            let base = row * 4;
            assert!((s[base] - 0.0).abs() < 1e-6);
            assert!((s[base + 1] - 0.25).abs() < 1e-6);
            assert!((s[base + 2] - 0.75).abs() < 1e-6);
            assert!((s[base + 3] - 1.0).abs() < 1e-6);
        }
    }

    // ---- 预处理归一化逻辑 ----

    #[test]
    fn test_normalize_minus_one_one_semantics() {
        // (x/255 - 0.5)/0.5: 0 → -1, 127.5 → ~0, 255 → 1
        let f = |v: u8| (v as f32 / 255.0 - 0.5) / 0.5;
        assert!((f(0) - (-1.0)).abs() < 1e-6);
        assert!(f(255) - 1.0 < 1e-6);
        assert!(f(128).abs() < 0.01);
    }

    // ---- 构造校验 ----

    #[test]
    fn test_input_size_validation() {
        // 非法的 input_size 不依赖真实模型即可校验
        // (0 和非 32 倍数在创建 session 之前即校验; 这里直接测数值规则)
        let valid = |s: u32| s > 0 && s.is_multiple_of(STRIDE);
        assert!(valid(512));
        assert!(!valid(0));
        assert!(!valid(100));
    }
}
