// ====================================================================
// 人脸对齐工具 — 仿射变换、关键点变换 (纯Rust实现)
// ====================================================================
// 基于标准 5 点对齐方案

/// 标准 5 点参考坐标 (112x112 图像)
pub const ARC_FACE_REFERENCE_112: [(f32, f32); 5] = [
    (38.2946, 51.6963), // 左眼
    (73.5318, 51.5014), // 右眼
    (56.0252, 71.7366), // 鼻子
    (41.5493, 92.3655), // 左嘴角
    (70.7299, 92.2041), // 右嘴角
];

/// 从源关键点到目标关键点估算相似变换矩阵 (2x3)
///
/// 使用最小二乘法求解缩放+旋转+平移
///
/// # 返回
/// 2x3 仿射变换矩阵
pub fn estimate_similarity_transform(
    src: &[(f32, f32)], // (N, 2) 源点集
    dst: &[(f32, f32)], // (N, 2) 目标点集
) -> [[f32; 3]; 2] {
    assert_eq!(
        src.len(),
        dst.len(),
        "Source and destination point counts must match"
    );
    let n = src.len();

    // 计算质心
    let src_cx: f32 = src.iter().map(|p| p.0).sum::<f32>() / n as f32;
    let src_cy: f32 = src.iter().map(|p| p.1).sum::<f32>() / n as f32;
    let dst_cx: f32 = dst.iter().map(|p| p.0).sum::<f32>() / n as f32;
    let dst_cy: f32 = dst.iter().map(|p| p.1).sum::<f32>() / n as f32;

    // 零均值化
    let src_centered: Vec<(f32, f32)> = src.iter().map(|p| (p.0 - src_cx, p.1 - src_cy)).collect();
    let dst_centered: Vec<(f32, f32)> = dst.iter().map(|p| (p.0 - dst_cx, p.1 - dst_cy)).collect();

    // 计算协方差矩阵项
    // a = s·cosθ, b = s·sinθ (Umeyama 相似变换闭式解)
    let mut a = 0.0f32;
    let mut b = 0.0f32;
    for i in 0..n {
        let (sx, sy) = src_centered[i];
        let (dx, dy) = dst_centered[i];
        a += sx * dx + sy * dy;
        b += sx * dy - sy * dx;
    }

    // 归一化
    let sum_sq: f32 = src_centered.iter().map(|p| p.0 * p.0 + p.1 * p.1).sum();
    if sum_sq < 1e-10 {
        // 退化为纯平移
        return [[1.0, 0.0, dst_cx - src_cx], [0.0, 1.0, dst_cy - src_cy]];
    }

    let a = a / sum_sq;
    let b = b / sum_sq;

    // 构建旋转缩放矩阵 [ [a, -b, tx], [b, a, ty] ] (R = [[cosθ,-sinθ],[sinθ,cosθ]])
    let tx = dst_cx - a * src_cx + b * src_cy;
    let ty = dst_cy - b * src_cx - a * src_cy;

    [[a, -b, tx], [b, a, ty]]
}

/// 使用仿射变换矩阵变换 2D 点集
///
/// # 参数
/// * `points` - (N, 2) 输入点集
/// * `matrix` - 2x3 仿射变换矩阵
///
/// # 返回
/// (N, 2) 变换后的点集
pub fn transform_points_2d(points: &[(f32, f32)], matrix: &[[f32; 3]; 2]) -> Vec<(f32, f32)> {
    points
        .iter()
        .map(|&(x, y)| {
            let new_x = matrix[0][0] * x + matrix[0][1] * y + matrix[0][2];
            let new_y = matrix[1][0] * x + matrix[1][1] * y + matrix[1][2];
            (new_x, new_y)
        })
        .collect()
}

/// 逆仿射变换矩阵
pub fn invert_affine(matrix: &[[f32; 3]; 2]) -> [[f32; 3]; 2] {
    let det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    if det.abs() < 1e-10 {
        return *matrix; // 奇异矩阵, 无法求逆
    }
    let inv_det = 1.0 / det;
    let a = matrix[0][0] * inv_det;
    let b = matrix[0][1] * inv_det;
    let c = matrix[1][0] * inv_det;
    let d = matrix[1][1] * inv_det;

    let tx = -(d * matrix[0][2] - b * matrix[1][2]);
    let ty = -(-c * matrix[0][2] + a * matrix[1][2]);

    [[d, -b, tx], [-c, a, ty]]
}

/// 使用仿射变换矩阵对图像进行 warp (最近邻)
///
/// 注意: 这是简化版 — 生产中建议使用 GPU 加速或 SIMD 优化
pub fn warp_affine_nearest(
    image: &[u8],
    w: u32,
    h: u32,
    out_w: u32,
    out_h: u32,
    matrix: &[[f32; 3]; 2],
) -> Vec<u8> {
    let inv = invert_affine(matrix);
    let mut out = vec![0u8; (out_w * out_h * 3) as usize];

    for y in 0..out_h {
        for x in 0..out_w {
            let src_x = inv[0][0] * x as f32 + inv[0][1] * y as f32 + inv[0][2];
            let src_y = inv[1][0] * x as f32 + inv[1][1] * y as f32 + inv[1][2];

            let sx = src_x.round() as i32;
            let sy = src_y.round() as i32;

            let out_idx = ((y * out_w + x) * 3) as usize;

            if sx >= 0 && sx < w as i32 && sy >= 0 && sy < h as i32 {
                let in_idx = ((sy as u32 * w + sx as u32) * 3) as usize;
                out[out_idx] = image[in_idx];
                out[out_idx + 1] = image[in_idx + 1];
                out[out_idx + 2] = image[in_idx + 2];
            }
            // else: 超出边界的像素保持为0 (黑色)
        }
    }

    out
}

/// 5点对齐: 用5点关键点估算相似变换 (对齐到 112x112 参考坐标)
///
/// # 参数
/// * `landmarks` - 5点关键点: 左眼, 右眼, 鼻, 左嘴角, 右嘴角
///
/// # 返回
/// (forward 2x3 正向变换矩阵, inverse 2x3 逆向变换矩阵)
pub fn estimate_norm_112(landmarks: &[(f32, f32); 5]) -> ([[f32; 3]; 2], [[f32; 3]; 2]) {
    let forward = estimate_similarity_transform(landmarks, &ARC_FACE_REFERENCE_112);
    let inverse = invert_affine(&forward);
    (forward, inverse)
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_estimate_similarity_matches_reference() {
        // 与参考实现 (Umeyama 最小二乘相似变换) 对齐:
        // 求解 dst = src @ (rotation*scale) + translation 的最小二乘
        // 相似变换, 与本实现的闭式解在数学上等价 (参考标准 5 点验证)
        let src = [
            (269.3800048828125f32, 269.29998779296875),
            (334.5899963378906, 271.35998535235998535),
            (314.0599975583789, 318.0199890136789),
            (271.3299865722656, 346.739990234375),
            (319.4599914550781, 348.4200134277344),
        ];
        let m = super::estimate_similarity_transform(&src, &super::ARC_FACE_REFERENCE_112);
        // 仿射变换参考值: [[0.5224, 0.0376, -113.2974], [-0.0376, 0.5224, -79.0983]]
        assert!((m[0][0] - 0.5224).abs() < 2e-3, "m00={}", m[0][0]);
        assert!((m[0][1] - 0.0376).abs() < 2e-3, "m01={}", m[0][1]);
        assert!((m[1][0] - (-0.0376)).abs() < 2e-3, "m10={}", m[1][0]);
        assert!((m[0][2] - (-113.2974)).abs() < 0.05, "tx={}", m[0][2]);
        assert!((m[1][2] - (-79.0983)).abs() < 0.05, "ty={}", m[1][2]);
    }

    use super::*;

    #[test]
    fn test_estimate_similarity_identity() {
        let src = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)];
        let dst = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)];
        let mat = estimate_similarity_transform(&src, &dst);

        // 应该接近单位矩阵
        assert!((mat[0][0] - 1.0).abs() < 0.01);
        assert!((mat[1][1] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_transform_points() {
        let points = [(10.0, 20.0)];
        let matrix = [[2.0, 0.0, 5.0], [0.0, 2.0, 10.0]];
        let result = transform_points_2d(&points, &matrix);
        assert_eq!(result[0], (25.0, 50.0));
    }

    #[test]
    fn test_estimate_norm_112_identity() {
        // 使用参考点本身作为输入, 应得到近似单位变换
        let (forward, inverse) = estimate_norm_112(&ARC_FACE_REFERENCE_112);

        for r in 0..2 {
            for c in 0..2 {
                let expected = if r == c { 1.0 } else { 0.0 };
                assert!(
                    (forward[r][c] - expected).abs() < 1e-3,
                    "forward[{}][{}]={}",
                    r,
                    c,
                    forward[r][c]
                );
                assert!(
                    (inverse[r][c] - expected).abs() < 1e-3,
                    "inverse[{}][{}]={}",
                    r,
                    c,
                    inverse[r][c]
                );
            }
        }
        // 平移项应接近 0
        assert!(forward[0][2].abs() < 1e-3);
        assert!(forward[1][2].abs() < 1e-3);
        assert!(inverse[0][2].abs() < 1e-3);
        assert!(inverse[1][2].abs() < 1e-3);
    }

    #[test]
    fn test_estimate_norm_112_maps_landmarks_to_reference() {
        // 缩放+平移的非退化5点经正向变换后应精确映射到参考点
        let s = 2.0f32;
        let (tx, ty) = (5.0f32, -3.0f32);
        let landmarks: Vec<(f32, f32)> = ARC_FACE_REFERENCE_112
            .iter()
            .map(|&(x, y)| (x / s - tx, y / s - ty))
            .collect();
        let (forward, _) = estimate_norm_112(&[
            landmarks[0],
            landmarks[1],
            landmarks[2],
            landmarks[3],
            landmarks[4],
        ]);
        let mapped = transform_points_2d(&landmarks, &forward);

        for (got, want) in mapped.iter().zip(ARC_FACE_REFERENCE_112.iter()) {
            assert!((got.0 - want.0).abs() < 1e-2, "x: {} vs {}", got.0, want.0);
            assert!((got.1 - want.1).abs() < 1e-2, "y: {} vs {}", got.1, want.1);
        }
    }
}
