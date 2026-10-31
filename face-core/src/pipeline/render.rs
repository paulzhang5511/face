// ====================================================================
// 结果可视化 — 检测框 / 关键点 / FaceMesh / 解析遮罩叠加
// ====================================================================

use crate::types::{DetectedFace, ParsingMask};
use image::{Rgb, RgbImage};

/// 解析标签颜色 (19 类) / Parsing label palette (19 classes)
const LABEL_COLORS: [[u8; 3]; 19] = [
    [0, 0, 0],       // background
    [128, 64, 128],  // skin
    [244, 35, 232],  // l_brow
    [35, 142, 107],  // r_brow
    [187, 128, 87],  // l_eye
    [146, 70, 48],   // r_eye
    [60, 20, 220],   // eye_g
    [0, 236, 236],   // l_ear
    [250, 170, 30],  // r_ear
    [0, 130, 180],   // ear_r
    [220, 80, 60],   // nose
    [0, 0, 66],      // mouth
    [170, 0, 184],   // u_lip
    [153, 153, 153], // l_lip
    [90, 120, 220],  // neck
    [220, 220, 0],   // neck_l
    [230, 0, 0],     // cloth
    [90, 0, 90],     // hair
    [30, 0, 255],    // hat
];

/// 绘制检测框 + 5点关键点 + 属性文本摘要
pub fn draw_faces(img: &RgbImage, faces: &[DetectedFace]) -> RgbImage {
    let mut canvas = img.clone();
    for face in faces {
        let (x1, y1, x2, y2) = face.bbox.to_int();
        let (x1, y1, x2, y2) = (
            x1.max(0) as u32,
            y1.max(0) as u32,
            x2.min(canvas.width() as i32 - 1).max(0) as u32,
            y2.min(canvas.height() as i32 - 1).max(0) as u32,
        );
        let color = Rgb([0, 255, 0]);
        // 边框 / border
        for x in x1..=x2 {
            canvas.put_pixel(x, y1, color);
            canvas.put_pixel(x, y2, color);
        }
        for y in y1..=y2 {
            canvas.put_pixel(x1, y, color);
            canvas.put_pixel(x2, y, color);
        }
        // 5 点关键点 / 5-pt landmarks
        if let Some(lm) = &face.landmarks {
            for p in lm.rows() {
                let px = p[[0]] as i32;
                let py = p[[1]] as i32;
                for dy in -2..=2 {
                    for dx in -2..=2 {
                        let cx = px + dx;
                        let cy = py + dy;
                        if cx >= 0
                            && cy >= 0
                            && (cx as u32) < canvas.width()
                            && (cy as u32) < canvas.height()
                        {
                            canvas.put_pixel(cx as u32, cy as u32, Rgb([255, 0, 0]));
                        }
                    }
                }
            }
        }
        // FaceMesh 稀疏绘制 (每隔 6 个点) / sparse FaceMesh dots
        if let Some(mesh) = &face.mesh_3d {
            for (i, p) in mesh.rows().into_iter().enumerate() {
                if i % 6 != 0 {
                    continue;
                }
                let px = p[[0]] as i32;
                let py = p[[1]] as i32;
                if px >= 0
                    && py >= 0
                    && (px as u32) < canvas.width()
                    && (py as u32) < canvas.height()
                {
                    canvas.put_pixel(px as u32, py as u32, Rgb([0, 255, 255]));
                }
            }
        }
    }
    canvas
}

/// 解析遮罩半透明叠加 / parsing mask alpha overlay
pub fn draw_parsing(img: &RgbImage, mask: &ParsingMask) -> RgbImage {
    let mut canvas = img.clone();
    if (mask.width, mask.height) != (canvas.width(), canvas.height()) {
        return canvas;
    }
    for y in 0..canvas.height() {
        for x in 0..canvas.width() {
            let label = mask.mask[[y as usize, x as usize]];
            if label == 0 {
                continue;
            }
            let c = LABEL_COLORS[label as usize % 19];
            let p = canvas.get_pixel_mut(x, y);
            let blended = [
                (p[0] as u32 * 55 + c[0] as u32 * 200) / 255,
                (p[1] as u32 * 55 + c[1] as u32 * 200) / 255,
                (p[2] as u32 * 55 + c[2] as u32 * 200) / 255,
            ];
            *p = Rgb([blended[0] as u8, blended[1] as u8, blended[2] as u8]);
        }
    }
    canvas
}

/// 纯色解析可视化 / solid-color parsing visualization
pub fn draw_parsing_solid(mask: &ParsingMask) -> RgbImage {
    let mut canvas = RgbImage::new(mask.width, mask.height);
    for y in 0..mask.height {
        for x in 0..mask.width {
            let label = mask.mask[[y as usize, x as usize]];
            if label != 0 {
                let c = LABEL_COLORS[label as usize % 19];
                canvas.put_pixel(x, y, Rgb(c));
            }
        }
    }
    canvas
}
