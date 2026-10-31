// ====================================================================
// 人脸匿名化
// ====================================================================
//
// 支持的 5 种模式:
//   - Blur / BlurStrong      -> 盒式模糊
//   - Pixelate / PixelateStrong -> 块平均像素化 (块大小为像素宽度)
//   - Cover                  -> 黑色矩形填充
//
// 实现说明:
//   1. 模糊采用迭代盒式模糊近似高斯模糊 (kernel 奇数半径)
//   2. 像素化以"块像素宽度"定义 (而非块数量)
//   3. 未实现椭圆羽化模糊 — 需要全图高斯混合, 超出纯算法模块范围

use image::{DynamicImage, GenericImageView, RgbImage};

use crate::types::BBox;

/// 强模糊固定核大小 (像素)
pub const STRONG_BLUR_KERNEL: u32 = 51;

/// 强像素化固定块宽度 (像素)
pub const STRONG_PIXELATE_BLOCK: u32 = 16;

/// 匿名化模式 — 共 5 种方法
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnonymizeMode {
    /// 盒式模糊, kernel 为奇数核大小 (像素)
    Blur { kernel: u32 },
    /// 块像素化, block 为块宽度 (像素)
    Pixelate { block: u32 },
    /// 强模糊 — 固定核 51
    BlurStrong,
    /// 强像素化 — 固定块宽 16
    PixelateStrong,
    /// 纯黑矩形覆盖
    Cover,
}

impl AnonymizeMode {
    /// 归一化核大小为奇数且 >= 3
    #[must_use]
    fn effective_kernel(kernel: u32) -> u32 {
        let k = kernel.max(3);
        if k.is_multiple_of(2) { k + 1 } else { k }
    }
}

/// 对图中每个 bbox 区域应用匿名化, 返回新图
///
/// bbox 会先 clamp 到图像边界; 空 bbox 列表时返回原图副本
pub fn anonymize_regions(img: &DynamicImage, boxes: &[BBox], mode: &AnonymizeMode) -> DynamicImage {
    let (img_w, img_h) = img.dimensions();
    let rgb = img.to_rgb8();
    let mut out =
        RgbImage::from_raw(img_w, img_h, rgb.into_raw()).expect("image buffer size mismatch");

    for bbox in boxes {
        // clamp 到图像边界
        let clamped = bbox.clamp(img_w as f32, img_h as f32);
        let x1 = clamped.x1 as u32;
        let y1 = clamped.y1 as u32;
        let x2 = (clamped.x2 as u32).min(img_w);
        let y2 = (clamped.y2 as u32).min(img_h);

        if x2 <= x1 || y2 <= y1 {
            continue;
        }

        match mode {
            AnonymizeMode::Cover => {
                for y in y1..y2 {
                    for x in x1..x2 {
                        out.put_pixel(x, y, image::Rgb([0u8, 0, 0]));
                    }
                }
            }
            AnonymizeMode::Blur { kernel } => {
                let k = AnonymizeMode::effective_kernel(*kernel);
                blur_region(&mut out, x1, y1, x2, y2, k);
            }
            AnonymizeMode::BlurStrong => {
                blur_region(&mut out, x1, y1, x2, y2, STRONG_BLUR_KERNEL);
            }
            AnonymizeMode::Pixelate { block } => {
                let b = (*block).max(1);
                pixelate_region(&mut out, x1, y1, x2, y2, b);
            }
            AnonymizeMode::PixelateStrong => {
                pixelate_region(&mut out, x1, y1, x2, y2, STRONG_PIXELATE_BLOCK);
            }
        }
    }

    DynamicImage::ImageRgb8(out)
}

/// 区域盒式模糊: 迭代水平/垂直前缀和, 等效单次盒式滤波
///
/// 采用盒式滤波近似高斯模糊 (工程简化)
fn blur_region(img: &mut RgbImage, x1: u32, y1: u32, x2: u32, y2: u32, kernel: u32) {
    let w = x2 - x1;
    let h = y2 - y1;
    let radius = (kernel / 2).min(w.max(h)); // 核心不超出区域

    // 提取区域
    let mut region = RgbImage::new(w, h);
    for dy in 0..h {
        for dx in 0..w {
            region.put_pixel(dx, dy, *img.get_pixel(x1 + dx, y1 + dy));
        }
    }

    // 盒式模糊 (水平 + 垂直各一次)
    let blurred = box_blur(&region, radius);
    for dy in 0..h {
        for dx in 0..w {
            img.put_pixel(x1 + dx, y1 + dy, *blurred.get_pixel(dx, dy));
        }
    }
}

/// 单通道/三通道联合盒式模糊 (半径 r), 边缘自动收缩窗口
fn box_blur(img: &RgbImage, r: u32) -> RgbImage {
    let (w, h) = img.dimensions();
    if r == 0 || w == 0 || h == 0 {
        let mut out = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                out.put_pixel(x, y, *img.get_pixel(x, y));
            }
        }
        return out;
    }

    // 每像素窗口 [x-r, x+r] (clamp 到边界), 用前缀和 O(1) 求和
    let mut hz = RgbImage::new(w, h);
    for y in 0..h {
        // 行前缀和: ps[k+1] = 前 k+1 个像素之和
        let mut ps = vec![[0i64; 3]; w as usize + 1];
        for x in 0..w {
            let p = img.get_pixel(x, y);
            for c in 0..3 {
                ps[x as usize + 1][c] = ps[x as usize][c] + p[c] as i64;
            }
        }
        for x in 0..w {
            let lo = x.saturating_sub(r) as usize;
            let hi = (x + r).min(w - 1) as usize;
            let cnt = (hi - lo + 1) as f32;
            let px = hz.get_pixel_mut(x, y);
            for c in 0..3 {
                let sum = ps[hi + 1][c] - ps[lo][c];
                px[c] = ((sum as f32 / cnt).round()).clamp(0.0, 255.0) as u8;
            }
        }
    }

    // 垂直同理
    let mut out = RgbImage::new(w, h);
    for x in 0..w {
        let mut ps = vec![[0i64; 3]; h as usize + 1];
        for y in 0..h {
            let p = hz.get_pixel(x, y);
            for c in 0..3 {
                ps[y as usize + 1][c] = ps[y as usize][c] + p[c] as i64;
            }
        }
        for y in 0..h {
            let lo = y.saturating_sub(r) as usize;
            let hi = (y + r).min(h - 1) as usize;
            let cnt = (hi - lo + 1) as f32;
            let px = out.get_pixel_mut(x, y);
            for c in 0..3 {
                let sum = ps[hi + 1][c] - ps[lo][c];
                px[c] = ((sum as f32 / cnt).round()).clamp(0.0, 255.0) as u8;
            }
        }
    }

    out
}

/// 区域像素化: 按 block 像素宽分块, 块内取平均后整块填充
fn pixelate_region(img: &mut RgbImage, x1: u32, y1: u32, x2: u32, y2: u32, block: u32) {
    let mut by = y1;
    while by < y2 {
        let mut bx = x1;
        while bx < x2 {
            let ex = (bx + block).min(x2);
            let ey = (by + block).min(y2);

            // 块内平均
            let mut sum = [0u64; 3];
            let mut n = 0u64;
            for y in by..ey {
                for x in bx..ex {
                    let p = img.get_pixel(x, y);
                    for c in 0..3 {
                        sum[c] += p[c] as u64;
                    }
                    n += 1;
                }
            }
            let mean = [(sum[0] / n) as u8, (sum[1] / n) as u8, (sum[2] / n) as u8];

            for y in by..ey {
                for x in bx..ex {
                    img.put_pixel(x, y, image::Rgb(mean));
                }
            }

            bx = ex;
        }
        by = ey_next(by, y2, block);
    }
}

/// 推进下一块的 y 起点 (避免内联闭包借用)
#[inline]
fn ey_next(by: u32, y2: u32, block: u32) -> u32 {
    (by + block).min(y2)
}

// ====================================================================
// 单元测试
// ====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// 构造纯色测试图
    fn solid_image(w: u32, h: u32, color: [u8; 3]) -> DynamicImage {
        let mut img = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                img.put_pixel(x, y, image::Rgb(color));
            }
        }
        DynamicImage::ImageRgb8(img)
    }

    #[test]
    fn test_pixelate_solid_region_unchanged() {
        // 纯色区域像素化后应保持不变
        let img = solid_image(64, 64, [200, 100, 50]);
        let boxes = [BBox::new(8.0, 8.0, 56.0, 56.0)];
        let out = anonymize_regions(&img, &boxes, &AnonymizeMode::Pixelate { block: 8 });
        assert_eq!(out.to_rgb8().as_raw(), img.to_rgb8().as_raw());
    }

    #[test]
    fn test_output_dimensions_match_input() {
        let img = solid_image(120, 80, [10, 20, 30]);
        let boxes = [BBox::new(10.0, 10.0, 60.0, 60.0)];
        for mode in [
            AnonymizeMode::Blur { kernel: 5 },
            AnonymizeMode::Pixelate { block: 4 },
            AnonymizeMode::BlurStrong,
            AnonymizeMode::PixelateStrong,
            AnonymizeMode::Cover,
        ] {
            let out = anonymize_regions(&img, &boxes, &mode);
            assert_eq!(out.dimensions(), (120, 80), "dims mismatch for {:?}", mode);
        }
    }

    #[test]
    fn test_cover_blacks_out_region() {
        let img = solid_image(32, 32, [255, 255, 255]);
        let boxes = [BBox::new(4.0, 4.0, 12.0, 12.0)];
        let out = anonymize_regions(&img, &boxes, &AnonymizeMode::Cover);
        let rgb = out.to_rgb8();
        assert_eq!(rgb.get_pixel(8, 8), &image::Rgb([0, 0, 0]));
        // 区域外不受影响
        assert_eq!(rgb.get_pixel(1, 1), &image::Rgb([255, 255, 255]));
    }

    #[test]
    fn test_bbox_clamped_to_image_bounds() {
        // 越界 bbox 应被 clamp 而非 panic
        let img = solid_image(40, 40, [0, 255, 0]);
        let boxes = [BBox::new(-10.0, -10.0, 100.0, 100.0)];
        let out = anonymize_regions(&img, &boxes, &AnonymizeMode::Cover);
        let rgb = out.to_rgb8();
        assert_eq!(rgb.get_pixel(0, 0), &image::Rgb([0, 0, 0]));
        assert_eq!(rgb.get_pixel(39, 39), &image::Rgb([0, 0, 0]));
    }

    #[test]
    fn test_empty_boxes_returns_copy() {
        let img = solid_image(16, 16, [1, 2, 3]);
        let out = anonymize_regions(&img, &[], &AnonymizeMode::BlurStrong);
        assert_eq!(out.to_rgb8().as_raw(), img.to_rgb8().as_raw());
    }

    #[test]
    fn test_pixelate_averages_blocks() {
        // 左半黑右半白, 单块覆盖全区域 -> 块平均值 ~127
        let mut img = RgbImage::new(16, 8);
        for y in 0..8 {
            for x in 0..16 {
                let v = if x < 8 { 0u8 } else { 255u8 };
                img.put_pixel(x, y, image::Rgb([v, v, v]));
            }
        }
        let dimg = DynamicImage::ImageRgb8(img);
        let boxes = [BBox::new(0.0, 0.0, 16.0, 8.0)];
        let out = anonymize_regions(&dimg, &boxes, &AnonymizeMode::Pixelate { block: 16 });
        let rgb = out.to_rgb8();
        for y in 0..8 {
            for x in 0..16 {
                let v = rgb.get_pixel(x, y)[0];
                assert!((120..=135).contains(&v), "pixel ({},{}) = {}", x, y, v);
            }
        }
    }

    #[test]
    fn test_blur_smooths_gradient() {
        // 模糊后边缘对比度应下降
        let mut img = RgbImage::new(32, 32);
        for y in 0..32 {
            for x in 0..32 {
                let v = if x < 16 { 0u8 } else { 255u8 };
                img.put_pixel(x, y, image::Rgb([v, v, v]));
            }
        }
        let dimg = DynamicImage::ImageRgb8(img);
        let boxes = [BBox::new(0.0, 0.0, 32.0, 32.0)];
        let out = anonymize_regions(&dimg, &boxes, &AnonymizeMode::Blur { kernel: 15 });
        let rgb = out.to_rgb8();
        // 中心列不再是非黑即白
        let mid = rgb.get_pixel(16, 16)[0];
        assert!(mid > 20 && mid < 235, "mid = {}", mid);
    }
}
