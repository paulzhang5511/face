// ====================================================================
// 图像处理工具 — 跨平台纯Rust实现 (基于 image crate)
// ====================================================================
// 替代 OpenCV 功能: 颜色转换、缩放、裁剪、仿射变换

use anyhow::{Context, Result};
use image::{DynamicImage, GenericImageView, ImageDecoder, Pixel, Rgb, RgbImage};
use log::trace;
use ndarray::Array3;

/// 将 image crate 的 RGB 图像转换为 ndarray (C, H, W) float32 [0.0, 1.0]
pub fn rgb_to_ndarray_chw(img: &RgbImage) -> Array3<f32> {
    let (w, h) = img.dimensions();
    let mut arr = Array3::<f32>::zeros((3, h as usize, w as usize));

    for (x, y, pixel) in img.enumerate_pixels() {
        let ch = pixel.channels();
        arr[[0, y as usize, x as usize]] = ch[0] as f32 / 255.0;
        arr[[1, y as usize, x as usize]] = ch[1] as f32 / 255.0;
        arr[[2, y as usize, x as usize]] = ch[2] as f32 / 255.0;
    }

    arr
}

/// 将 ndarray (C, H, W) float32 转换回 RGB 图像
pub fn ndarray_chw_to_rgb(arr: &Array3<f32>) -> Result<RgbImage> {
    let shape = arr.shape();
    if shape.len() != 3 || shape[0] != 3 {
        anyhow::bail!("Expected array shape (3, H, W), got {:?}", shape);
    }
    let h = shape[1] as u32;
    let w = shape[2] as u32;
    let mut img = RgbImage::new(w, h);

    for y in 0..h {
        for x in 0..w {
            let r = (arr[[0, y as usize, x as usize]] * 255.0)
                .round()
                .clamp(0.0, 255.0) as u8;
            let g = (arr[[1, y as usize, x as usize]] * 255.0)
                .round()
                .clamp(0.0, 255.0) as u8;
            let b = (arr[[2, y as usize, x as usize]] * 255.0)
                .round()
                .clamp(0.0, 255.0) as u8;
            img.put_pixel(x, y, Rgb([r, g, b]));
        }
    }

    Ok(img)
}

/// 高质量图像缩放 (Lanczos3 插值)
pub fn resize_image(img: &DynamicImage, target_w: u32, target_h: u32) -> DynamicImage {
    trace!(
        "Resizing image from {:?} to {}x{}",
        img.dimensions(),
        target_w,
        target_h
    );
    img.resize_exact(target_w, target_h, image::imageops::FilterType::Lanczos3)
}

/// 快速图像缩放 (最近邻插值 — 用于遮罩/标签图像)
pub fn resize_mask(mask: &[u8], src_w: u32, src_h: u32, target_w: u32, target_h: u32) -> Vec<u8> {
    let mut result = vec![0u8; (target_w * target_h) as usize];
    let x_scale = src_w as f64 / target_w as f64;
    let y_scale = src_h as f64 / target_h as f64;

    for y in 0..target_h {
        for x in 0..target_w {
            let src_x = ((x as f64) * x_scale) as u32;
            let src_y = ((y as f64) * y_scale) as u32;
            let src_x = src_x.min(src_w - 1);
            let src_y = src_y.min(src_h - 1);
            result[(y * target_w + x) as usize] = mask[(src_y * src_w + src_x) as usize];
        }
    }

    result
}

/// 从图像中裁剪人脸区域 (带边界检查)
pub fn crop_face(img: &DynamicImage, x1: i32, y1: i32, x2: i32, y2: i32) -> Result<DynamicImage> {
    let (img_w, img_h) = img.dimensions();
    let x1 = x1.max(0) as u32;
    let y1 = y1.max(0) as u32;
    let x2 = x2.min(img_w as i32).max(0) as u32;
    let y2 = y2.min(img_h as i32).max(0) as u32;

    if x2 <= x1 || y2 <= y1 {
        anyhow::bail!(
            "Invalid crop region: ({},{}) to ({},{}) on image {}x{}",
            x1,
            y1,
            x2,
            y2,
            img_w,
            img_h
        );
    }

    Ok(img.crop_imm(x1, y1, x2 - x1, y2 - y1))
}

/// 从原始像素字节加载图像
pub fn load_image_from_bytes(bytes: &[u8]) -> Result<DynamicImage> {
    image::load_from_memory(bytes).context("Failed to decode image from bytes")
}

/// 加载图像并应用 EXIF 方向
pub fn load_image_with_exif(path: &str) -> Result<DynamicImage> {
    let bytes = std::fs::read(path).context("Failed to read image file")?;
    let cursor = std::io::Cursor::new(&bytes);
    let orientation = image::ImageReader::new(cursor.clone())
        .with_guessed_format()
        .context("Failed to detect image format")?
        .into_decoder()
        .and_then(|mut d| d.orientation())?;
    let mut img = image::ImageReader::new(cursor)
        .with_guessed_format()
        .context("Failed to detect image format")?
        .decode()
        .context("Failed to decode image")?;
    img.apply_orientation(orientation);
    Ok(img)
}

/// 将图像编码为 JPEG 字节
pub fn encode_jpeg(img: &DynamicImage, _quality: u8) -> Result<Vec<u8>> {
    let mut buf = std::io::Cursor::new(Vec::new());
    img.write_to(&mut buf, image::ImageFormat::Jpeg)
        .context("Failed to encode JPEG")?;
    Ok(buf.into_inner())
}

/// 将图像编码为 PNG 字节
pub fn encode_png(img: &DynamicImage) -> Result<Vec<u8>> {
    let mut buf = std::io::Cursor::new(Vec::new());
    img.write_to(&mut buf, image::ImageFormat::Png)
        .context("Failed to encode PNG")?;
    Ok(buf.into_inner())
}

/// 归一化: (x - mean) / std (针对 CHW 格式)
pub fn normalize_chw(arr: &mut Array3<f32>, mean: &[f32; 3], std: &[f32; 3]) {
    let h = arr.shape()[1];
    let w = arr.shape()[2];
    for c in 0..3 {
        let m = mean[c];
        let s = std[c];
        for y in 0..h {
            for x in 0..w {
                arr[[c, y, x]] = (arr[[c, y, x]] - m) / s;
            }
        }
    }
}

/// 归一化到 [-1, 1]: (x - 127.5) / 127.5
pub fn normalize_minus_one_one(arr: &mut Array3<f32>) {
    for val in arr.iter_mut() {
        *val = (*val - 0.5) * 2.0;
    }
}

/// letterbox: 保持纵横比缩放并铺到 target_w x target_h 画布, 空白处为黑色
///
/// # 返回
/// (画布图像, 缩放比, 水平填充, 垂直填充)
pub fn letterbox(img: &DynamicImage, target_w: u32, target_h: u32) -> (RgbImage, f32, f32, f32) {
    let (img_w, img_h) = img.dimensions();
    // 等比缩放 + 左上角锚定 (不居中)
    let im_ratio = img_h as f32 / img_w as f32;
    let model_ratio = target_h as f32 / target_w as f32;
    let (new_w, new_h) = if im_ratio > model_ratio {
        let nh = target_h;
        let nw = ((nh as f32 / im_ratio) as u32).max(1);
        (nw.min(target_w), nh)
    } else {
        let nw = target_w;
        let nh = ((nw as f32 * im_ratio) as u32).max(1);
        (nw, nh.min(target_h))
    };

    let scale = new_h as f32 / img_h as f32;
    // 双线性插值缩放 (半像素中心对齐语义)
    let resized = resize_bilinear_cv2(&img.to_rgb8(), new_w, new_h);

    let mut canvas = RgbImage::from_pixel(target_w, target_h, Rgb([0u8, 0, 0]));
    image::imageops::overlay(&mut canvas, &resized, 0, 0);

    (canvas, scale, 0.0, 0.0)
}

/// 双线性插值缩放: 半像素中心 + 11bit 定点系数
pub fn resize_bilinear_cv2(img: &RgbImage, new_w: u32, new_h: u32) -> RgbImage {
    let (w, h) = img.dimensions();
    let sx = w as f32 / new_w as f32;
    let sy = h as f32 / new_h as f32;
    const SHIFT: u32 = 11;
    const ONE: u32 = 1 << SHIFT; // 2048
    // 预计算每列/行的索引与定点系数
    let mut xs = vec![(0usize, 0usize, 0u32, 0u32); new_w as usize];
    for x in 0..new_w as usize {
        let fx = (x as f32 + 0.5) * sx - 0.5;
        let x0 = fx.floor() as i64;
        let dx = fx - x0 as f32;
        let c1 = (dx * ONE as f32).round() as u32;
        let c0 = ONE - c1;
        let x0c = x0.clamp(0, (w - 1) as i64) as usize;
        let x1c = (x0 + 1).clamp(0, (w - 1) as i64) as usize;
        xs[x] = (x0c, x1c, c0, c1);
    }
    let mut ys = vec![(0usize, 0usize, 0u32, 0u32); new_h as usize];
    for y in 0..new_h as usize {
        let fy = (y as f32 + 0.5) * sy - 0.5;
        let y0 = fy.floor() as i64;
        let dy = fy - y0 as f32;
        let c1 = (dy * ONE as f32).round() as u32;
        let c0 = ONE - c1;
        let y0c = y0.clamp(0, (h - 1) as i64) as usize;
        let y1c = (y0 + 1).clamp(0, (h - 1) as i64) as usize;
        ys[y] = (y0c, y1c, c0, c1);
    }

    let mut out = RgbImage::new(new_w, new_h);
    for (y, &(y0c, y1c, b0, b1)) in ys.iter().enumerate() {
        for (x, &(x0c, x1c, a0, a1)) in xs.iter().enumerate() {
            let p00 = img.get_pixel(x0c as u32, y0c as u32).channels();
            let p10 = img.get_pixel(x1c as u32, y0c as u32).channels();
            let p01 = img.get_pixel(x0c as u32, y1c as u32).channels();
            let p11 = img.get_pixel(x1c as u32, y1c as u32).channels();
            let mut px = [0u8; 3];
            for c in 0..3 {
                let r0 = p00[c] as u32 * a0 + p10[c] as u32 * a1;
                let r1 = p01[c] as u32 * a0 + p11[c] as u32 * a1;
                let v = r0 * b0 + r1 * b1;
                px[c] = ((v + (1 << (2 * SHIFT - 1))) >> (2 * SHIFT)) as u8;
            }
            out.put_pixel(x as u32, y as u32, Rgb(px));
        }
    }
    out
}

/// 双线性采样单个像素 (用于 warp); 越界返回黑色
pub fn bilinear_sample(img: &RgbImage, x: f32, y: f32) -> Rgb<u8> {
    let (w, h) = img.dimensions();
    if w == 0 || h == 0 {
        return Rgb([0u8, 0, 0]);
    }

    let x0 = x.floor();
    let y0 = y.floor();
    let fx = x - x0;
    let fy = y - y0;

    let xi0 = x0 as i64;
    let yi0 = y0 as i64;
    let xi1 = xi0 + 1;
    let yi1 = yi0 + 1;

    // 越界直接返回黑色
    if xi1 < 0 || yi1 < 0 || xi0 >= w as i64 || yi0 >= h as i64 {
        return Rgb([0u8, 0, 0]);
    }

    let get = |px: i64, py: i64| -> [f32; 3] {
        // 越界邻域视为边界值黑色
        if px < 0 || py < 0 || px >= w as i64 || py >= h as i64 {
            return [0.0, 0.0, 0.0];
        }
        let c = img.get_pixel(px as u32, py as u32).channels();
        [c[0] as f32, c[1] as f32, c[2] as f32]
    };

    let p00 = get(xi0, yi0);
    let p10 = get(xi1, yi0);
    let p01 = get(xi0, yi1);
    let p11 = get(xi1, yi1);

    let mut out = [0u8; 3];
    for c in 0..3 {
        let top = p00[c] * (1.0 - fx) + p10[c] * fx;
        let bottom = p01[c] * (1.0 - fx) + p11[c] * fx;
        let value = top * (1.0 - fy) + bottom * fy;
        out[c] = value.round().clamp(0.0, 255.0) as u8;
    }
    Rgb(out)
}

/// warpAffine 双线性版 (边界外填黑色)
///
/// `matrix` 为 输出->输入 的正向映射由调用者提供, 内部用逆矩阵逐输出像素采样
pub fn warp_affine_bilinear(
    img: &RgbImage,
    matrix: &[[f32; 3]; 2],
    out_w: u32,
    out_h: u32,
) -> RgbImage {
    let inv = crate::utils::alignment::invert_affine(matrix);
    let mut out = RgbImage::from_pixel(out_w, out_h, Rgb([0u8, 0, 0]));

    // 像素中心约定: 正向与逆向变换的半像素偏移相互抵消
    for y in 0..out_h {
        for x in 0..out_w {
            let src_x = inv[0][0] * x as f32 + inv[0][1] * y as f32 + inv[0][2];
            let src_y = inv[1][0] * x as f32 + inv[1][1] * y as f32 + inv[1][2];
            let pixel = bilinear_sample(img, src_x, src_y);
            out.put_pixel(x, y, pixel);
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_warp_affine_identity() {
        // 单位矩阵 warp 应还原原图 (4x4)
        let mut img = RgbImage::new(4, 4);
        for y in 0..4 {
            for x in 0..4 {
                img.put_pixel(x, y, Rgb([(x * 40) as u8, (y * 50) as u8, 128]));
            }
        }

        let identity = [[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let warped = warp_affine_bilinear(&img, &identity, 4, 4);

        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(
                    img.get_pixel(x, y),
                    warped.get_pixel(x, y),
                    "mismatch at ({},{})",
                    x,
                    y
                );
            }
        }
    }

    #[test]
    fn test_warp_affine_out_of_bounds_is_black() {
        // 平移超出原图范围的区域应为黑色
        let img = RgbImage::from_pixel(4, 4, Rgb([200u8, 100, 50]));
        // 整体向右平移2像素
        let matrix = [[1.0f32, 0.0, 2.0], [0.0, 1.0, 0.0]];
        let warped = warp_affine_bilinear(&img, &matrix, 6, 4);

        assert_eq!(warped.get_pixel(0, 0), &Rgb([0u8, 0, 0]));
        assert_eq!(warped.get_pixel(1, 0), &Rgb([0u8, 0, 0]));
        assert_eq!(warped.get_pixel(2, 0), &Rgb([200u8, 100, 50]));
        assert_eq!(warped.get_pixel(5, 3), &Rgb([200u8, 100, 50]));
    }

    #[test]
    fn test_letterbox_wide_image_anchors_top_left() {
        // 宽图 letterbox 到正方形: 左上角锚定, 下方为黑色
        let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(20, 10, Rgb([10u8, 20, 30])));
        let (canvas, scale, pad_x, pad_y) = letterbox(&img, 20, 20);

        assert!((scale - 1.0).abs() < 1e-6);
        assert!(pad_x.abs() < 1e-6);
        assert!(pad_y.abs() < 1e-6);

        // 顶部行保留原图像素, 底部填充为黑色
        for x in 0..20 {
            assert_eq!(canvas.get_pixel(x, 0), &Rgb([10u8, 20, 30]));
            assert_eq!(canvas.get_pixel(x, 10), &Rgb([0u8, 0, 0]));
            assert_eq!(canvas.get_pixel(x, 19), &Rgb([0u8, 0, 0]));
        }
    }

    #[test]
    fn test_letterbox_tall_image_anchors_top_left() {
        // 高图 letterbox 到正方形: 左上角锚定, 右侧为黑色
        let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(8, 16, Rgb([60u8, 120, 180])));
        let (canvas, scale, pad_x, pad_y) = letterbox(&img, 16, 16);

        assert!((scale - 1.0).abs() < 1e-6);
        assert!(pad_x.abs() < 1e-6);
        assert!(pad_y.abs() < 1e-6);

        for y in 0..16 {
            assert_eq!(canvas.get_pixel(0, y), &Rgb([60u8, 120, 180]));
            assert_eq!(canvas.get_pixel(8, y), &Rgb([0u8, 0, 0]));
            assert_eq!(canvas.get_pixel(15, y), &Rgb([0u8, 0, 0]));
        }
    }

    #[test]
    fn test_bilinear_sample_out_of_bounds() {
        let img = RgbImage::from_pixel(4, 4, Rgb([1u8, 2, 3]));
        assert_eq!(bilinear_sample(&img, -1.0, -1.0), Rgb([0u8, 0, 0]));
        assert_eq!(bilinear_sample(&img, 10.0, 10.0), Rgb([0u8, 0, 0]));
        assert_eq!(bilinear_sample(&img, 0.0, 0.0), Rgb([1u8, 2, 3]));
    }
}
