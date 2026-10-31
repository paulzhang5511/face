// ====================================================================
// 头部姿态估计 — 几何解算 + ONNX ResNet18
// ====================================================================
//
// 两条路径:
//   1. estimate_from_mesh: 纯几何 — 用 FaceMesh 468 点的 3D 坐标
//      做共面 PnP 等价解算 (迭代最小二乘解, 无需相机内参文件)
//   2. HeadPoseNet: 6D 旋转表示的 ResNet18 图像模型, 输出 (1,3,3) 旋转矩阵
//
// 模型来源:

use anyhow::Result;
use image::{DynamicImage, GenericImageView};
use log::{debug, info, trace};
use ndarray::{Array2, Array4};
use ort::value::TensorRef;
use rayon::prelude::*;

use crate::inference::InferenceSession;
use crate::types::{BBox, DetectedFace, HeadPoseResult};
use crate::utils::image_utils::resize_bilinear_cv2;

/// HeadPose ResNet18 输入尺寸
const HEADPOSE_INPUT_SIZE: u32 = 224;

/// ImageNet 均值 (RGB)
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
/// ImageNet 标准差 (RGB)
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

// --------------------------------------------------------------------
// 几何头姿解算
// --------------------------------------------------------------------

/// 3x3 旋转矩阵 (行优先)
type Rot3 = [[f32; 3]; 3];

/// 旋转矩阵 → 欧拉角 (ZYX 约定, 弧度)
///
/// 保持弧度不转度
///
/// 返回 (pitch_x, yaw_y, roll_z)
pub fn rotation_matrix_to_euler(r: &Rot3) -> (f32, f32, f32) {
    let sy = (r[0][0] * r[0][0] + r[1][0] * r[1][0]).sqrt();
    let singular = sy < 1e-6;

    let pitch = if singular {
        (-r[1][2]).atan2(r[1][1])
    } else {
        r[2][1].atan2(r[2][2])
    };
    let yaw = (-r[2][0]).atan2(sy);
    let roll = if singular {
        0.0
    } else {
        r[1][0].atan2(r[0][0])
    };

    (pitch, yaw, roll)
}

/// 3x3 矩阵乘法 (测试辅助)
#[cfg(test)]
fn mat3_mul(a: &Rot3, b: &Rot3) -> Rot3 {
    let mut r = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    r
}

/// 由 (pitch, yaw, roll) 构造 ZYX 旋转矩阵 R = Rz(roll)·Ry(yaw)·Rx(pitch)
///
/// 与 `rotation_matrix_to_euler` 的分解约定严格互逆 (标准 ZYX 矩阵)
#[cfg(test)]
fn euler_to_rotation(pitch: f32, yaw: f32, roll: f32) -> Rot3 {
    let (sp, cp) = (pitch.sin(), pitch.cos());
    let (sy, cy) = (yaw.sin(), yaw.cos());
    let (sr, cr) = (roll.sin(), roll.cos());
    [
        [cr * cy, cr * sy * sp - sr * cp, sr * sp + cr * sy * cp],
        [sr * cy, sr * sy * sp + cr * cp, sr * sy * cp - cr * sp],
        [-sy, cy * sp, cy * cp],
    ]
}

/// 高斯牛顿法求解最小二乘线性方程组 (正规方程 + 高斯消元)
///
/// 求解 A·x = b (A: n x n 对称正定, 维度由调用方给定); 奇异时返回 None
fn solve_linear(a: &[Vec<f64>], b: &[f64], n_par: usize) -> Option<Vec<f64>> {
    // 增广矩阵 [A | b] (f64 避免病态数值)
    let mut m: Vec<Vec<f64>> = (0..n_par)
        .map(|i| {
            let mut row = vec![0.0f64; n_par + 1];
            row[..n_par].copy_from_slice(&a[i][..n_par]);
            row[n_par] = b[i];
            row
        })
        .collect();

    for col in 0..n_par {
        // 列主元
        let piv = (col..n_par)
            .max_by(|&x, &y| m[x][col].abs().partial_cmp(&m[y][col].abs()).unwrap())
            .unwrap_or(col);
        m.swap(col, piv);
        if m[col][col].abs() < 1e-12 {
            return None;
        }
        for r in (col + 1)..n_par {
            let factor = m[r][col] / m[col][col];
            for c in col..=n_par {
                m[r][c] -= factor * m[col][c];
            }
        }
    }

    let mut x = vec![0.0f64; n_par];
    for i in (0..n_par).rev() {
        let mut acc = m[i][n_par];
        for j in (i + 1)..n_par {
            acc -= m[i][j] * x[j];
        }
        x[i] = acc / m[i][i];
    }
    Some(x)
}

/// 归一化 3x3 矩阵的 Frobenius 范数
#[allow(dead_code)]
fn frobenius(r: &Rot3) -> f32 {
    (0..3)
        .flat_map(|a| (0..3).map(move |b| r[a][b] * r[a][b]))
        .sum::<f32>()
        .sqrt()
}

/// 相机内参近似 (像素焦距)
///
/// 常见近似 f ≈ 图像宽度 (水平 FOV ≈ 60°)
fn focal_length(img_w: u32) -> f32 {
    let w = img_w.max(1) as f32;
    // 常用近似: f = w
    w
}

/// 从 FaceMesh 468 点 3D 坐标几何解算头部姿态 (共面 PnP 等价)
///
/// 思路 (共面 PnP 的迭代解等价实现):
///   1. 以标准正面人脸关键点的 2D 均值模板作为“模型点” X_model (z=0 共面,
///      以鼻尖为原点, 单位: 相对尺度)
///   2. 旋转候选 R(θ) 作用于模型点, 再经针孔投影 (焦距 f ≈ max(w,h)) 与观测
///      2D 点对齐
///   3. 高斯牛顿最小化重投影误差, 联合优化 (pitch, yaw, roll, tx, ty, tz, scale)
///
/// 参数:
///   * `landmarks_3d` - (468, 3); x/y 为像素坐标, z 为相对深度
///   * `img_w`, `img_h` - 原图尺寸 (用于估计焦距)
///
pub fn estimate_from_mesh(landmarks_3d: &Array2<f32>, img_w: u32, img_h: u32) -> HeadPoseResult {
    let n = landmarks_3d.shape()[0].min(468);
    if n < 468 {
        // 语义锚点索引不足, 退化为无旋转
        return HeadPoseResult {
            pitch: 0.0,
            yaw: 0.0,
            roll: 0.0,
        };
    }

    let f = focal_length(img_w.max(img_h));
    let cx = img_w as f32 / 2.0;
    let cy = img_h as f32 / 2.0;

    // 1. 标准正面模板: 左右眼外/内角、鼻尖、嘴中心 (共面 z=0)
    let model_pts: [[f32; 3]; 6] = [
        [-0.060, -0.035, 0.0], // 左眼外角
        [-0.025, -0.035, 0.0], // 左眼内角
        [0.025, -0.035, 0.0],  // 右眼内角
        [0.060, -0.035, 0.0],  // 右眼外角
        [0.0, 0.030, 0.0],     // 鼻尖
        [0.0, 0.105, 0.0],     // 嘴中心
    ];

    // 对应的 FaceMesh 468 点索引 (标准 FaceMesh 布局)
    let idx = [
        33usize, // 左眼外角
        133,     // 左眼内角
        362,     // 右眼内角
        263,     // 右眼外角
        1,       // 鼻尖
        13,      // 上唇内侧中心
    ];

    // 观测的归一化平面坐标 (像素 → 归一化)
    let obs_uv: [(f32, f32); 6] = {
        let mut arr = [(0.0f32, 0.0f32); 6];
        for (k, &i) in idx.iter().enumerate() {
            arr[k] = (
                (landmarks_3d[[i, 0]] - cx) / f,
                (landmarks_3d[[i, 1]] - cy) / f,
            );
        }
        arr
    };

    // 2. 初始 scale: 观测眼外角距 / 模板眼外角距
    let tpl_eye = ((model_pts[3][0] - model_pts[0][0]).powi(2)
        + (model_pts[3][1] - model_pts[0][1]).powi(2))
    .sqrt();
    let obs_eye =
        ((obs_uv[3].0 - obs_uv[0].0).powi(2) + (obs_uv[3].1 - obs_uv[0].1).powi(2)).sqrt();
    let scale0 = if tpl_eye > 1e-9 {
        obs_eye / tpl_eye
    } else {
        1.0
    };

    // 参数: [pitch, yaw, roll, tx, ty, scale] — tz 固定为 1.0 消除尺度规范退化
    //       (s 与 t 同比缩放时重投影不变, 7 参数下正规方程奇异)
    const N_PAR: usize = 6;
    let mut p: [f64; N_PAR] = [0.0, 0.0, 0.0, 0.0, 0.0, scale0 as f64];
    let tz: f64 = 1.0;

    // 完整残差实现 — 全程 f64 (f32 残差 + 数值雅可比会因舍入噪声使正规方程病态)
    let residual = |p: &[f64; N_PAR]| -> Vec<f64> {
        let (pitch, yaw, roll) = (p[0], p[1], p[2]);
        let (tx, ty, s) = (p[3], p[4], p[5]);
        let (sp, cp) = (pitch.sin(), pitch.cos());
        let (sy, cy) = (yaw.sin(), yaw.cos());
        let (sr, cr) = (roll.sin(), roll.cos());
        // R = Rz(roll)·Ry(yaw)·Rx(pitch) (f64 版, 同 euler_to_rotation)
        let r: [[f64; 3]; 3] = [
            [cr * cy, cr * sy * sp - sr * cp, sr * sp + cr * sy * cp],
            [sr * cy, sr * sy * sp + cr * cp, sr * sy * cp - cr * sp],
            [-sy, cy * sp, cy * cp],
        ];
        let mut res = Vec::with_capacity(model_pts.len() * 2);
        for (k, m) in model_pts.iter().enumerate() {
            let (m0, m1) = (m[0] as f64, m[1] as f64);
            let x = r[0][0] * m0 * s + r[0][1] * m1 * s + tx;
            let y = r[1][0] * m0 * s + r[1][1] * m1 * s + ty;
            let z = r[2][0] * m0 * s + r[2][1] * m1 * s + tz;
            if z.abs() < 1e-9 {
                res.push(1e6);
                res.push(1e6);
            } else {
                res.push((x / z) - obs_uv[k].0 as f64);
                res.push((y / z) - obs_uv[k].1 as f64);
            }
        }
        res
    };

    // 3. 高斯牛顿迭代
    const MAX_ITERS: usize = 50;
    for _ in 0..MAX_ITERS {
        let r0 = residual(&p);

        // 数值雅可比 (中心差分精度足够, 步长对角度/平移/尺度通用)
        let eps = 1e-6f64;
        let mut j_cols: Vec<Vec<f64>> = Vec::with_capacity(N_PAR);
        for j in 0..N_PAR {
            let mut p2 = p;
            p2[j] += eps;
            let r1 = residual(&p2);
            j_cols.push(
                r1.iter()
                    .zip(r0.iter())
                    .map(|(a, b)| (a - b) / eps)
                    .collect(),
            );
        }

        // 正规方程 A·δ = -J^T·r
        let mut a: Vec<Vec<f64>> = vec![vec![0.0f64; N_PAR]; N_PAR];
        let mut b = vec![0.0f64; N_PAR];
        for row in 0..N_PAR {
            for (col, jb) in j_cols.iter().enumerate() {
                a[row][col] = j_cols[row].iter().zip(jb.iter()).map(|(x, y)| x * y).sum();
            }
            b[row] = -j_cols[row]
                .iter()
                .zip(r0.iter())
                .map(|(x, y)| x * y)
                .sum::<f64>();
        }

        let delta = match solve_linear(&a, &b, N_PAR) {
            Some(d) => d,
            None => break,
        };
        for i in 0..N_PAR {
            p[i] += delta[i];
        }
        let step: f64 = delta.iter().map(|d| d * d).sum();
        if step < 1e-14 {
            break;
        }
    }

    // 4. 欧拉角即优化变量 (ZYX 约定)
    let (pitch, yaw, roll) = (p[0] as f32, p[1] as f32, p[2] as f32);

    trace!(
        "HeadPose(mesh): pitch={:.3} yaw={:.3} roll={:.3} rad ({} pts, f={:.0})",
        pitch, yaw, roll, n, f
    );

    HeadPoseResult { pitch, yaw, roll }
}

// --------------------------------------------------------------------
// HeadPoseNet (ResNet18, 6D 旋转表示)
// --------------------------------------------------------------------

/// 头部姿态图像模型 (ResNet18, 6D 旋转表示)
///
/// - 预处理: bbox 中心裁剪正方形 → resize 224x224 → RGB → [0,1] →
///   ImageNet 归一化 → NCHW
/// - 输出: (1, 3, 3) 旋转矩阵 → ZYX 欧拉角
///
/// 模型来源 / Model source:
pub struct HeadPoseNet {
    session: InferenceSession,
    input_size: u32,
}

impl HeadPoseNet {
    /// 创建 HeadPoseNet
    /// HeadPose ResNet18: 224x224 输入
    pub fn from_session(session: InferenceSession) -> Result<Self> {
        let input_size = HEADPOSE_INPUT_SIZE;
        info!("HeadPoseNet: input_size={}x{}", input_size, input_size);
        Ok(Self {
            session,
            input_size,
        })
    }

    /// 预测头部姿态
    pub fn predict(&mut self, image: &DynamicImage, face: &DetectedFace) -> Result<HeadPoseResult> {
        let input_array = self.preprocess(image, &face.bbox)?;

        let input = TensorRef::from_array_view(&input_array)?;
        let output_name = self.session.session().outputs()[0].name().to_string();
        let outputs = self.session.session_mut().run(ort::inputs![input])?;

        let (_shape, rot_ref) = outputs
            .get(output_name.as_str())
            .ok_or_else(|| anyhow::anyhow!("HeadPose model output '{}' not found", output_name))?
            .try_extract_tensor::<f32>()?;
        let rot_data: Vec<f32> = rot_ref.to_vec();
        drop(outputs);

        let result = parse_headpose_output(&rot_data)?;

        trace!(
            "HeadPoseNet: pitch={:.3} yaw={:.3} roll={:.3} rad",
            result.pitch, result.yaw, result.roll
        );
        Ok(result)
    }

    /// 批量预测头部姿态: 逐脸预处理后拼 (N,3,H,W) 单次推理
    ///
    /// 返回按输入顺序对齐的结果; 预处理失败的脸为 None
    pub fn predict_batch(
        &mut self,
        image: &DynamicImage,
        faces: &[&DetectedFace],
    ) -> Result<Vec<Option<HeadPoseResult>>> {
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
                    log::debug!("HeadPoseNet: face {} preprocessing failed: {}", i, e);
                }
            }
        }
        let mut results: Vec<Option<HeadPoseResult>> = vec![None; faces.len()];
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
        let output_name = self.session.session().outputs()[0].name().to_string();
        let outputs = self.session.session_mut().run(ort::inputs![tensor])?;
        let (_shape, rot_ref) = outputs
            .get(output_name.as_str())
            .ok_or_else(|| anyhow::anyhow!("HeadPose model output '{}' not found", output_name))?
            .try_extract_tensor::<f32>()?;
        let rot_data: Vec<f32> = rot_ref.to_vec();
        drop(outputs);

        let dim = rot_data.len() / n;
        for (b, &i) in valid.iter().enumerate() {
            let row = &rot_data[b * dim..(b + 1) * dim];
            if let Ok(r) = parse_headpose_output(row) {
                results[i] = Some(r);
            }
        }
        Ok(results)
    }

    /// 预处理: bbox 中心裁剪正方形 → resize → ImageNet 归一化 → NCHW
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
                "HeadPoseNet: bbox ({},{})-({},{}) does not overlap the {}x{} image",
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

        debug!(
            "HeadPoseNet: preprocessed {}x{} crop -> {}x{} NCHW",
            x2 - x1,
            y2 - y1,
            size,
            size
        );

        Ok(arr)
    }
}

/// 解析 HeadPoseNet 输出
///
/// 输入为展平的 (1, 3, 3) 旋转矩阵; 输出 ZYX 欧拉角 (弧度)
pub fn parse_headpose_output(rot_flat: &[f32]) -> Result<HeadPoseResult> {
    if rot_flat.len() < 9 {
        anyhow::bail!(
            "HeadPose output too short: {} floats, expected >= 9",
            rot_flat.len()
        );
    }
    let r: Rot3 = [
        [rot_flat[0], rot_flat[1], rot_flat[2]],
        [rot_flat[3], rot_flat[4], rot_flat[5]],
        [rot_flat[6], rot_flat[7], rot_flat[8]],
    ];
    let (pitch, yaw, roll) = rotation_matrix_to_euler(&r);
    Ok(HeadPoseResult { pitch, yaw, roll })
}

// --------------------------------------------------------------------
// 单元测试
// --------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// 构造合成 FaceMesh 3D 点: 标准正面人脸模板经给定旋转后投影
    fn synthetic_mesh(yaw_deg: f32, pitch_deg: f32, roll_deg: f32) -> Array2<f32> {
        let (yaw, pitch, roll) = (
            yaw_deg.to_radians(),
            pitch_deg.to_radians(),
            roll_deg.to_radians(),
        );

        // 绕轴旋转矩阵 (相机坐标系: x 右, y 下, z 前)
        let rotx = |t: f32| -> Rot3 {
            [
                [1.0, 0.0, 0.0],
                [0.0, t.cos(), -t.sin()],
                [0.0, t.sin(), t.cos()],
            ]
        };
        let roty = |t: f32| -> Rot3 {
            [
                [t.cos(), 0.0, t.sin()],
                [0.0, 1.0, 0.0],
                [-t.sin(), 0.0, t.cos()],
            ]
        };
        let rotz = |t: f32| -> Rot3 {
            [
                [t.cos(), -t.sin(), 0.0],
                [t.sin(), t.cos(), 0.0],
                [0.0, 0.0, 1.0],
            ]
        };
        let mul = |a: &Rot3, b: &Rot3| -> Rot3 {
            let mut r = [[0.0f32; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
                }
            }
            r
        };

        // R = Rz(roll) · Ry(yaw) · Rx(pitch) — 与 euler_to_rotation 一致
        let r = mul(&rotz(roll), &mul(&roty(yaw), &rotx(pitch)));

        // 468 个模板点: 在脸部区域内近似均匀分布 (含 6 个语义锚点)
        let mut pts = Vec::with_capacity(468 * 3);
        // 焦距必须与 estimate_from_mesh 的近似一致: f = max(img_w, img_h) = max(640,480)
        let f = 640.0f32;
        let cx = 320.0f32;
        let cy = 240.0f32;

        // 语义锚点必须与 estimate_from_mesh 的模板一一对应
        let anchors = [
            [-0.060, -0.035, 0.0], // 33 左眼外角
            [-0.025, -0.035, 0.0], // 133 左眼内角
            [0.025, -0.035, 0.0],  // 362 右眼内角
            [0.060, -0.035, 0.0],  // 263 右眼外角
            [0.0, 0.030, 0.0],     // 1 鼻尖
            [0.0, 0.105, 0.0],     // 13 嘴中心
        ];
        let anchor_idx = [33usize, 133, 362, 263, 1, 13];

        let mut data = vec![[0.0f32; 3]; 468];
        for i in 0..468 {
            let (mx, my): (f32, f32) = if let Some(pos) = anchor_idx.iter().position(|&a| a == i) {
                (anchors[pos][0], anchors[pos][1])
            } else {
                // 其余点: 网格散布在 [-0.08, 0.08] x [-0.05, 0.12]
                let gx = ((i % 24) as f32 - 11.5) * 0.007;
                let gy = ((i / 24) as f32 - 9.5) * 0.014;
                (gx, gy)
            };

            let mz = 0.0f32; // 模型平面 z=0 (共面)
            let px = r[0][0] * mx + r[0][1] * my + r[0][2] * mz;
            let py = r[1][0] * mx + r[1][1] * my + r[1][2] * mz;
            let pz = r[2][0] * mx + r[2][1] * my + r[2][2] * mz + 1.0; // 相机距离 1.0

            // 针孔投影 → 像素坐标; z 列为归一化深度
            data[i] = [f * px / pz + cx, f * py / pz + cy, f * (1.0 - 1.0 / pz)];
        }

        for p in data {
            pts.extend_from_slice(&p);
        }
        Array2::from_shape_vec((468, 3), pts).unwrap()
    }

    #[test]
    fn test_rotation_matrix_to_euler_identity() {
        let r: Rot3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let (p, y, r_) = rotation_matrix_to_euler(&r);
        assert!((p - 0.0).abs() < 1e-6);
        assert!((y - 0.0).abs() < 1e-6);
        assert!((r_ - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_rotation_matrix_to_euler_zyx() {
        // 构造 ZYX 欧拉角对应的旋转矩阵, 验证能恢复
        let (pitch, yaw, roll) = (0.15f32, -0.35f32, 0.25f32);
        // R = Rz(roll) * Ry(yaw) * Rx(pitch) — 与 euler_to_rotation 一致
        let rx = [
            [1.0, 0.0, 0.0],
            [0.0, pitch.cos(), -pitch.sin()],
            [0.0, pitch.sin(), pitch.cos()],
        ];
        let ry = [
            [yaw.cos(), 0.0, yaw.sin()],
            [0.0, 1.0, 0.0],
            [-yaw.sin(), 0.0, yaw.cos()],
        ];
        let rz = [
            [roll.cos(), -roll.sin(), 0.0],
            [roll.sin(), roll.cos(), 0.0],
            [0.0, 0.0, 1.0],
        ];
        let r = mat3_mul(&mat3_mul(&rz, &ry), &rx);
        let (p2, y2, r2) = rotation_matrix_to_euler(&r);
        assert!((p2 - pitch).abs() < 1e-4, "pitch: {} vs {}", p2, pitch);
        assert!((y2 - yaw).abs() < 1e-4, "yaw: {} vs {}", y2, yaw);
        assert!((r2 - roll).abs() < 1e-4, "roll: {} vs {}", r2, roll);

        // euler_to_rotation 必须与手写矩阵一致 (严格互逆)
        let r3 = euler_to_rotation(pitch, yaw, roll);
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (r[i][j] - r3[i][j]).abs() < 1e-6,
                    "M[{}][{}]: {} vs {}",
                    i,
                    j,
                    r[i][j],
                    r3[i][j]
                );
            }
        }
    }

    #[test]
    fn test_estimate_from_mesh_frontal() {
        // 正面人脸 → yaw/pitch/roll ≈ 0
        let mesh = synthetic_mesh(0.0, 0.0, 0.0);
        let r = estimate_from_mesh(&mesh, 640, 480);
        assert!(
            r.yaw.abs() < 0.05,
            "frontal yaw should be ~0, got {}",
            r.yaw
        );
        assert!(
            r.pitch.abs() < 0.05,
            "frontal pitch should be ~0, got {}",
            r.pitch
        );
        assert!(
            r.roll.abs() < 0.05,
            "frontal roll should be ~0, got {}",
            r.roll
        );
    }

    #[test]
    fn test_estimate_from_mesh_yaw_30() {
        // 绕 Y 轴旋转 30° → yaw ≈ ±0.52 rad
        let yaw30 = 30.0f32.to_radians();
        let mesh = synthetic_mesh(30.0, 0.0, 0.0);
        let r = estimate_from_mesh(&mesh, 640, 480);
        assert!(
            (r.yaw - yaw30).abs() < 0.08,
            "yaw should be ~0.52, got {}",
            r.yaw
        );
        assert!(r.pitch.abs() < 0.08, "pitch should be ~0, got {}", r.pitch);
        assert!(r.roll.abs() < 0.08, "roll should be ~0, got {}", r.roll);
    }

    #[test]
    fn test_estimate_from_mesh_pitch_and_roll() {
        let pitch20 = 20.0f32.to_radians();
        let mesh = synthetic_mesh(0.0, 20.0, 0.0);
        let r = estimate_from_mesh(&mesh, 640, 480);
        assert!(
            (r.pitch - pitch20).abs() < 0.1,
            "pitch should be ~{}, got {}",
            pitch20,
            r.pitch
        );

        let roll15 = 15.0f32.to_radians();
        let mesh = synthetic_mesh(0.0, 0.0, 15.0);
        let r = estimate_from_mesh(&mesh, 640, 480);
        assert!(
            (r.roll - roll15).abs() < 0.1,
            "roll should be ~{}, got {}",
            roll15,
            r.roll
        );
    }

    #[test]
    fn test_parse_headpose_output() {
        // 单位阵 → 全零
        let out = parse_headpose_output(&[
            1.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.0, 0.0, 1.0,
        ])
        .unwrap();
        assert!(out.pitch.abs() < 1e-6);
        assert!(out.yaw.abs() < 1e-6);
        assert!(out.roll.abs() < 1e-6);

        // 已知小角度旋转: roll=0.3 (Rz), pitch=0.1 (Rx)
        let (sp, cp) = (0.1f32.sin(), 0.1f32.cos());
        let (sr, cr) = (0.3f32.sin(), 0.3f32.cos());
        // R = Rz(0.3) * Rx(0.1)
        let r = [
            cr,
            -sr * cp,
            sr * sp, //
            sr,
            cr * cp,
            -cr * sp, //
            0.0,
            sp,
            cp,
        ];
        let out = parse_headpose_output(&r).unwrap();
        assert!((out.pitch - 0.1).abs() < 1e-4);
        assert!(out.yaw.abs() < 1e-6);
        assert!((out.roll - 0.3).abs() < 1e-4);

        // 长度不足报错
        assert!(parse_headpose_output(&[0.1, 0.2, 0.3]).is_err());
    }

    #[test]
    fn test_parse_headpose_output_degrees_note() {
        // parse_headpose_output 输出弧度 (原始实现输出度数)
        let r = [
            0.0, -1.0, 0.0, // 90° roll
            1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0,
        ];
        let out = parse_headpose_output(&r).unwrap();
        let half_pi = std::f32::consts::FRAC_PI_2;
        assert!((out.roll - half_pi).abs() < 1e-4);
    }
}
