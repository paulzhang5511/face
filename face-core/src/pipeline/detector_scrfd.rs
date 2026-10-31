// ====================================================================
// SCRFD 人脸检测器 — 高效高精度人脸检测
// ====================================================================
//
// 模型: SCRFD (Sample and Computation Redistribution for Efficient Face Detection)
//       变体: scrfd_500m_kps / scrfd_10g_kps (带 5 点关键点)
//       - 输入: 640x640 RGB, 归一化到 [-1, 1]  ((px - 127.5) / 127.5)
//       - 输出: 9 个张量 = 3 stride x {score, bbox, kps}
//         (scrfd_500m_kps 实测 shape: score (N,1), bbox (N,4), kps (N,10),
//          N 按 stride 8/16/32 分别为 12800/3200/800, 共 16800 锚点)
//
// 解码与后处理关键约定:
//   1. 锚点中心 = grid * stride (无 +0.5 偏移), 每位置 2 个锚点 (num_anchors=2),
//      640 输入共 16800 个 — 已用真实 scrfd_500m_kps.onnx 实测验证:
//      grid*stride 解码出的人脸框与图像内容对齐, +0.5 变体则整体偏移出画布。
//   2. score 输出已经过图内 Sigmoid (ONNX 图含 Sigmoid 节点),
//      可直接与阈值比较, 此处不再重复施加 sigmoid。
//   3. bbox/kps 为 ltrb 距离, 解码前乘以对应 stride (distance2bbox/distance2kps)。
//   4. NMS 为标准贪心丢弃式 (抑制而非融合,
//      IoU > threshold 的候选被丢弃; 与 BlazeFace 的加权融合 NMS 不同)。
//   5. 5 点关键点顺序 (landmark order):
//      与通用 5 点对齐模板一致 — 满足对齐接口契约:
//      recognition / quality / XSeg 等对齐下游可直接消费本检测器的 landmarks。

use anyhow::{Result, bail};
use image::{DynamicImage, GenericImageView};
use log::{debug, info, trace};
use ndarray::Array2;
use ort::value::TensorRef;

use crate::inference::InferenceSession;
use crate::types::{BBox, DetectedFace};
use crate::utils::common::compute_iou;
use crate::utils::image_utils::letterbox;

/// 默认输入尺寸
const DEFAULT_INPUT_SIZE: u32 = 640;

/// FPN 特征步长
const DEFAULT_STRIDES: [usize; 3] = [8, 16, 32];

/// 每个特征位置的锚点数
const NUM_ANCHORS: usize = 2;

/// 借用形态的模型输出 (零拷贝: 直接引用 ORT 输出缓冲区)
///
/// 兼容 (N, C) 与 (1, N, C) 两种导出布局 (rank-2 / rank-3 with batch dim 1)
#[derive(Debug)]
struct BorrowedOutput<'a> {
    /// 行数 (锚点数或锚点组数)
    n: usize,
    /// 每行维度 (1/2=score, 4=bbox, 10=kps)
    dims: usize,
    /// 展平数据 (借用, 不拷贝)
    data: &'a [f32],
}

/// 输出分组结果: 三组输出索引, 每组按行数降序 (= stride 升序 8,16,32)
#[derive(Debug, Clone)]
struct OutputGroups {
    scores: Vec<usize>,
    bboxes: Vec<usize>,
    kpss: Vec<usize>,
}

/// SCRFD 检测器 — 高精度人脸检测, 输出 5 点对齐关键点
///
/// `use_kps = true` 时输出 (5, 2) landmarks, 顺序为
/// left_eye, right_eye, nose, left_mouth, right_mouth,
/// 可直接供 recognition / quality 等对齐流水线使用;
/// `use_kps = false` 时 landmarks 为 None, 仅输出 bbox。
pub struct ScrfdDetector {
    session: InferenceSession,
    confidence_threshold: f32,
    nms_threshold: f32,
    /// 输入边长 (正方形)
    input_size: u32,
    /// FPN 步长
    strides: Vec<usize>,
    /// 是否解码 5 点关键点
    use_kps: bool,
    /// 预生成锚点中心 (16800, 2), 按 stride 升序、行优先 (y 外层, x 内层)、
    /// 每位置重复 NUM_ANCHORS 次
    anchor_centers: Array2<f32>,
    /// 每个 stride 的锚点数量 (与 anchor_centers 前缀分段对应)
    stride_anchor_counts: Vec<usize>,
    /// 每个 stride 在 anchor_centers 中的起始偏移
    stride_anchor_offsets: Vec<usize>,
}

impl ScrfdDetector {
    /// 从已加载的 ONNX 会话创建 SCRFD 检测器 (输入 640x640, 步长 [8,16,32])
    pub fn from_session(
        session: InferenceSession,
        confidence_threshold: f32,
        nms_threshold: f32,
        use_kps: bool,
    ) -> Result<Self> {
        info!(
            "SCRFD: confidence_threshold={}, nms_threshold={}, use_kps={}",
            confidence_threshold, nms_threshold, use_kps
        );

        let strides: Vec<usize> = DEFAULT_STRIDES.to_vec();
        let (anchor_centers, stride_anchor_counts, stride_anchor_offsets) =
            generate_anchor_tables(DEFAULT_INPUT_SIZE, &strides);

        debug!(
            "SCRFD: generated {} anchor centers for strides {:?} (counts {:?})",
            anchor_centers.shape()[0],
            strides,
            stride_anchor_counts
        );

        Ok(Self {
            session,
            confidence_threshold,
            nms_threshold,
            input_size: DEFAULT_INPUT_SIZE,
            strides,
            use_kps,
            anchor_centers,
            stride_anchor_counts,
            stride_anchor_offsets,
        })
    }

    /// 检测图像中的所有人脸
    pub fn detect(&mut self, image: &DynamicImage) -> Result<Vec<DetectedFace>> {
        let (img_w, img_h) = image.dimensions();
        debug!(
            "SCRFD: detecting faces in {}x{} image (input {}x{})",
            img_w, img_h, self.input_size, self.input_size
        );

        // 1. 预处理: letterbox 到 640x640 + 归一化 [-1, 1] + NCHW
        let (canvas, scale, pad_x, pad_y) = letterbox(image, self.input_size, self.input_size);
        let input_array = preprocess(&canvas);

        // 2. ONNX 推理 + 零拷贝后处理:
        //    SessionOutputs 持有 session 的 &mut, 借用期内解码全部输出,
        //    避免逐输出 to_vec() 拷贝 (~1MB/帧)
        let faces = {
            let input = TensorRef::from_array_view(&input_array)?;
            let session = self.session.session_mut();
            let outputs = session.run(ort::inputs![input])?;
            // 先收集 ValueRef (借用 outputs), 再逐个提取数据视图,
            // 保证切片生命周期覆盖整个解码阶段
            let values: Vec<_> = outputs.iter().collect();
            let raws: Vec<BorrowedOutput> = values
                .iter()
                .map(|(_name, value)| {
                    let (shape, data) = value.try_extract_tensor::<f32>()?;
                    let (n, dims) = parse_output_shape(shape)?;
                    Ok::<_, anyhow::Error>(BorrowedOutput { n, dims, data })
                })
                .collect::<Result<_>>()?;

            debug!(
                "SCRFD inference: {} outputs, shapes {:?}",
                raws.len(),
                raws.iter().map(|r| (r.n, r.dims)).collect::<Vec<_>>()
            );

            // 分组 → 逐 stride 解码 → NMS → unletterbox
            // (字段级借用: strides/anchors 与 session 互不冲突)
            postprocess(
                &raws,
                &self.strides,
                &self.anchor_centers,
                &self.stride_anchor_counts,
                &self.stride_anchor_offsets,
                self.confidence_threshold,
                self.nms_threshold,
                self.use_kps,
                scale,
                pad_x,
                pad_y,
                img_w,
                img_h,
            )?
        };

        info!(
            "SCRFD: detected {} face(s) in {}x{} image",
            faces.len(),
            img_w,
            img_h
        );

        Ok(faces)
    }
}

/// 后处理: 解码候选、标准 NMS、恢复原图坐标
///
/// 以自由函数 + 显式参数形式实现, 以便在 SessionOutputs 存活期间
/// (session 可变借用未释放时) 借用 ScrfdDetector 的其余字段完成解码。
#[allow(clippy::too_many_arguments)]
fn postprocess(
    raws: &[BorrowedOutput],
    strides: &[usize],
    anchor_centers: &Array2<f32>,
    stride_anchor_counts: &[usize],
    stride_anchor_offsets: &[usize],
    confidence_threshold: f32,
    nms_threshold: f32,
    use_kps: bool,
    scale: f32,
    pad_x: f32,
    pad_y: f32,
    img_w: u32,
    img_h: u32,
) -> Result<Vec<DetectedFace>> {
    let shapes: Vec<(usize, usize)> = raws.iter().map(|r| (r.n, r.dims)).collect();
    let groups = group_outputs(&shapes)?;

    let num_strides = strides.len();
    if groups.scores.len() != num_strides || groups.bboxes.len() != num_strides {
        bail!(
            "SCRFD: expected {} score/bbox outputs, got {}/{}",
            num_strides,
            groups.scores.len(),
            groups.bboxes.len()
        );
    }
    let kps_raws: Option<Vec<&BorrowedOutput>> = if use_kps {
        if groups.kpss.len() != num_strides {
            bail!(
                "SCRFD: use_kps=true but found {} kps outputs (expected {})",
                groups.kpss.len(),
                num_strides
            );
        }
        Some(groups.kpss.iter().map(|&i| &raws[i]).collect())
    } else {
        None
    };

    // 候选: (score, canvas bbox, canvas kps)
    let mut candidates: Vec<(f32, [f32; 4], Option<[[f32; 2]; 5]>)> = Vec::new();

    for gi in 0..num_strides {
        let stride = strides[gi] as f32;
        let count = stride_anchor_counts[gi];
        let offset = stride_anchor_offsets[gi];

        let score_raw = &raws[groups.scores[gi]];
        // score 兼容 (N,1) 与 (locations,2) 布局: 总元素数必须等于锚点数
        if score_raw.data.len() != count {
            bail!(
                "SCRFD: stride {} score output has {} elements, expected {}",
                strides[gi],
                score_raw.data.len(),
                count
            );
        }

        let bbox_raw = &raws[groups.bboxes[gi]];
        if bbox_raw.dims != 4 || bbox_raw.n != count {
            bail!(
                "SCRFD: stride {} bbox output ({}, {}), expected ({}, 4)",
                strides[gi],
                bbox_raw.n,
                bbox_raw.dims,
                count
            );
        }

        let kps_raw = kps_raws.as_ref().and_then(|ks| ks.get(gi).copied());
        if let Some(k) = kps_raw {
            if k.dims != 10 || k.n != count {
                bail!(
                    "SCRFD: stride {} kps output ({}, {}), expected ({}, 10)",
                    strides[gi],
                    k.n,
                    k.dims,
                    count
                );
            }
        }

        for r in 0..count {
            // score 已经过图内 Sigmoid, 直接比较即可
            let score = score_raw.data[r];
            if score < confidence_threshold {
                continue;
            }

            let a = offset + r;
            let cx = anchor_centers[[a, 0]];
            let cy = anchor_centers[[a, 1]];

            // bbox: ltrb 距离 × stride, 加/减锚点中心
            let d = [
                bbox_raw.data[r * 4] * stride,
                bbox_raw.data[r * 4 + 1] * stride,
                bbox_raw.data[r * 4 + 2] * stride,
                bbox_raw.data[r * 4 + 3] * stride,
            ];
            let bbox = distance2bbox(cx, cy, &d);

            // kps: 5 点, 每点 (dx, dy) × stride, 加锚点中心
            let kps = kps_raw.map(|k| {
                let mut out = [[0.0f32; 2]; 5];
                for p in 0..5 {
                    out[p][0] = cx + k.data[r * 10 + p * 2] * stride;
                    out[p][1] = cy + k.data[r * 10 + p * 2 + 1] * stride;
                }
                out
            });

            candidates.push((score, bbox, kps));
        }
    }

    // 标准 NMS: 丢弃而非融合
    let dets: Vec<(f32, f32, f32, f32, f32)> = candidates
        .iter()
        .map(|(s, b, _)| (b[0], b[1], b[2], b[3], *s))
        .collect();
    let keep = standard_nms(&dets, nms_threshold);

    // unletterbox 回原图坐标并 clamp
    let mut faces = Vec::with_capacity(keep.len());
    for &i in &keep {
        let (score, bbox, kps) = &candidates[i];

        let x1 = clamp_f32((bbox[0] - pad_x) / scale, 0.0, img_w as f32);
        let y1 = clamp_f32((bbox[1] - pad_y) / scale, 0.0, img_h as f32);
        let x2 = clamp_f32((bbox[2] - pad_x) / scale, 0.0, img_w as f32);
        let y2 = clamp_f32((bbox[3] - pad_y) / scale, 0.0, img_h as f32);

        // 5 点关键点: (5, 2), 顺序 left_eye/right_eye/nose/left_mouth/right_mouth
        let landmarks = kps.map(|k| {
            let pts: Vec<f32> = k
                .iter()
                .flat_map(|[kx, ky]| {
                    let px = clamp_f32((kx - pad_x) / scale, 0.0, img_w as f32);
                    let py = clamp_f32((ky - pad_y) / scale, 0.0, img_h as f32);
                    vec![px, py]
                })
                .collect();
            Array2::from_shape_vec((5, 2), pts).expect("5x2 landmark buffer")
        });

        trace!(
            "SCRFD: face bbox=({:.1},{:.1},{:.1},{:.1}) score={:.3} landmarks={}",
            x1,
            y1,
            x2,
            y2,
            score,
            landmarks.is_some()
        );

        faces.push(DetectedFace {
            bbox: BBox::new(x1, y1, x2, y2),
            confidence: *score,
            landmarks,
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

    Ok(faces)
}

/// 生成锚点中心表及每 stride 的数量/偏移
///
/// 每层 `fm = input_size / stride` 个网格, 中心 = (x, y) * stride
/// (行优先: y 外层, x 内层), 每位置重复 `num_anchors` 次。
/// 640 输入 + strides [8,16,32] + num_anchors=2 → (80²+40²+20²)×2 = 16800。
fn generate_anchor_tables(
    input_size: u32,
    strides: &[usize],
) -> (Array2<f32>, Vec<usize>, Vec<usize>) {
    let total: usize = strides
        .iter()
        .map(|&s| {
            let fm = input_size as usize / s;
            fm * fm * NUM_ANCHORS
        })
        .sum();

    let mut flat: Vec<f32> = Vec::with_capacity(total * 2);
    let mut counts = Vec::with_capacity(strides.len());
    let mut offsets = Vec::with_capacity(strides.len());

    for &stride in strides {
        offsets.push(flat.len() / 2);
        let fm = input_size as usize / stride;
        let sf = stride as f32;
        for gy in 0..fm {
            for gx in 0..fm {
                let cx = gx as f32 * sf;
                let cy = gy as f32 * sf;
                for _ in 0..NUM_ANCHORS {
                    flat.push(cx);
                    flat.push(cy);
                }
            }
        }
        counts.push(fm * fm * NUM_ANCHORS);
    }

    let table = Array2::from_shape_vec((total, 2), flat).expect("even-length anchor buffer");
    (table, counts, offsets)
}

/// 预处理: RgbImage → (1, 3, H, W), 归一化 (px - 127.5) / 127.5
fn preprocess(canvas: &image::RgbImage) -> ndarray::Array4<f32> {
    let (w, h) = canvas.dimensions();
    let mut arr = ndarray::Array4::<f32>::zeros((1, 3, h as usize, w as usize));
    // 模型按 BGR 训练; image crate 加载为 RGB, 需交换通道
    // NCHW 平面布局: 三个连续通道平面, 每平面 H*W 连续像素 — 扁平化写避免逐点下标
    let hw = (h as usize) * (w as usize);
    let plane = arr.as_slice_mut().expect("Array4 contiguous");
    let (b_plane, rest) = plane.split_at_mut(hw);
    let (g_plane, r_plane) = rest.split_at_mut(hw);
    let src = canvas.as_raw();
    for i in 0..hw {
        let px = &src[i * 3..i * 3 + 3];
        b_plane[i] = (px[2] as f32 - 127.5) / 127.5;
        g_plane[i] = (px[1] as f32 - 127.5) / 127.5;
        r_plane[i] = (px[0] as f32 - 127.5) / 127.5;
    }
    arr
}

/// 解析输出张量 shape 为 (rows, dims), 兼容 (N, C) 与 (1, N, C)
fn parse_output_shape(shape: &[i64]) -> Result<(usize, usize)> {
    match shape {
        [n, c] if *n > 0 && *c > 0 => Ok((*n as usize, *c as usize)),
        [b, n, c] if *b == 1 && *n > 0 && *c > 0 => Ok((*n as usize, *c as usize)),
        _ => bail!("SCRFD: unsupported output shape {:?}", shape),
    }
}

/// 按 (rows, dims) 将全部输出分为 score/bbox/kps 三组
///
/// 判别规则 (兼容多种导出布局):
///   - dims 1 或 2 → score (scrfd_500m_kps 实测为 (N,1); 部分导出为 (N,2))
///   - dims 4      → bbox (ltrb 距离)
///   - dims 10     → kps (5 点 × 2 坐标)
/// 每组内部按 rows 降序排序 — rows 随 stride 增大而减小, 因此降序即
/// stride 升序 (8, 16, 32), 与锚点表分段顺序一致。
fn group_outputs(shapes: &[(usize, usize)]) -> Result<OutputGroups> {
    let mut scores: Vec<usize> = Vec::new();
    let mut bboxes: Vec<usize> = Vec::new();
    let mut kpss: Vec<usize> = Vec::new();

    for (i, &(_rows, dims)) in shapes.iter().enumerate() {
        match dims {
            1 | 2 => scores.push(i),
            4 => bboxes.push(i),
            10 => kpss.push(i),
            d => bail!("SCRFD: unexpected output dims {} at index {}", d, i),
        }
    }

    if scores.is_empty() || bboxes.is_empty() {
        bail!(
            "SCRFD: expected score and bbox outputs, got {} score / {} bbox",
            scores.len(),
            bboxes.len()
        );
    }
    if scores.len() != bboxes.len() {
        bail!(
            "SCRFD: score/bbox output count mismatch ({} vs {})",
            scores.len(),
            bboxes.len()
        );
    }

    let by_rows_desc = |idx: &mut Vec<usize>| {
        idx.sort_by(|&a, &b| shapes[b].0.cmp(&shapes[a].0));
    };
    by_rows_desc(&mut scores);
    by_rows_desc(&mut bboxes);
    by_rows_desc(&mut kpss);

    Ok(OutputGroups {
        scores,
        bboxes,
        kpss,
    })
}

/// distance2bbox: ltrb 距离解码 (负坐标下限截断到 0)
#[inline]
fn distance2bbox(cx: f32, cy: f32, d: &[f32; 4]) -> [f32; 4] {
    [
        (cx - d[0]).max(0.0),
        (cy - d[1]).max(0.0),
        (cx + d[2]).max(0.0),
        (cy + d[3]).max(0.0),
    ]
}

/// 标准贪心 NMS — 丢弃而非融合
///
/// 输入 dets 每行为 (x1, y1, x2, y2, score); 返回按得分降序保留的索引。
/// IoU > threshold 的后续候选被抑制 (即抑制严格大于阈值的重叠)。
fn standard_nms(dets: &[(f32, f32, f32, f32, f32)], threshold: f32) -> Vec<usize> {
    let n = dets.len();
    if n == 0 {
        return Vec::new();
    }

    // 按得分降序
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        dets[b]
            .4
            .partial_cmp(&dets[a].4)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut suppressed = vec![false; n];
    let mut keep = Vec::new();

    for &i in &order {
        if suppressed[i] {
            continue;
        }
        keep.push(i);
        for &j in &order {
            if i == j || suppressed[j] {
                continue;
            }
            let iou = compute_iou(
                (dets[i].0, dets[i].1, dets[i].2, dets[i].3),
                (dets[j].0, dets[j].1, dets[j].2, dets[j].3),
            );
            if iou > threshold {
                suppressed[j] = true;
            }
        }
    }

    keep
}

/// 裁剪值到范围 (复用 utils::common 的 clamp 语义)
#[inline]
fn clamp_f32(val: f32, min: f32, max: f32) -> f32 {
    if val < min {
        min
    } else if val > max {
        max
    } else {
        val
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anchor_generation_count_640() {
        // 640 输入, strides [8,16,32], 每位置 2 锚点 → 16800
        let (anchors, counts, offsets) = generate_anchor_tables(640, &[8, 16, 32]);

        assert_eq!(anchors.shape(), &[16800, 2]);
        assert_eq!(counts, vec![12800, 3200, 800]);
        assert_eq!(offsets, vec![0, 12800, 16000]);

        // stride 8 段: 行优先 (y 外层, x 内层), 中心 = grid * stride
        assert_eq!(anchors[[0, 0]], 0.0); // (x=0, y=0)
        assert_eq!(anchors[[0, 1]], 0.0);
        assert_eq!(anchors[[1, 0]], 0.0); // 同位置的第 2 个锚点 (重复)
        assert_eq!(anchors[[1, 1]], 0.0);
        assert_eq!(anchors[[2, 0]], 8.0); // (x=1, y=0)
        assert_eq!(anchors[[2, 1]], 0.0);
        assert_eq!(anchors[[160, 0]], 0.0); // (x=0, y=1): 行优先偏移 2*80
        assert_eq!(anchors[[160, 1]], 8.0);

        // stride 16 段起点
        assert_eq!(anchors[[12800, 0]], 0.0);
        assert_eq!(anchors[[12800, 1]], 0.0);
        assert_eq!(anchors[[12802, 0]], 16.0); // (x=1, y=0) at stride 16

        // stride 32 段起点
        assert_eq!(anchors[[16000, 0]], 0.0);
        assert_eq!(anchors[[16000, 1]], 0.0);
        assert_eq!(anchors[[16002, 0]], 32.0); // (x=1, y=0) at stride 32

        // 尾部锚点: stride 32, (x=19, y=19)
        assert_eq!(anchors[[16799, 0]], 19.0 * 32.0);
        assert_eq!(anchors[[16799, 1]], 19.0 * 32.0);
    }

    #[test]
    fn test_distance2bbox_decode() {
        // 常规解码: ltrb 距离加/减锚点中心
        let box_ = distance2bbox(100.0, 100.0, &[10.0, 20.0, 30.0, 40.0]);
        assert_eq!(box_, [90.0, 80.0, 130.0, 140.0]);

        // 负坐标下限截断到 0
        let clamped = distance2bbox(2.0, 3.0, &[5.0, 6.0, 1.0, 1.0]);
        assert_eq!(clamped, [0.0, 0.0, 3.0, 4.0]);

        // 零距离 → 退化为点
        let point = distance2bbox(50.0, 60.0, &[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(point, [50.0, 60.0, 50.0, 60.0]);
    }

    #[test]
    fn test_parse_output_shape() {
        // rank-2 (N, C) 与 rank-3 (1, N, C) 两种布局
        assert_eq!(parse_output_shape(&[12800, 1]).unwrap(), (12800, 1));
        assert_eq!(parse_output_shape(&[12800, 4]).unwrap(), (12800, 4));
        assert_eq!(parse_output_shape(&[1, 12800, 10]).unwrap(), (12800, 10));
        assert_eq!(parse_output_shape(&[1, 6400, 2]).unwrap(), (6400, 2));

        // 非法形状应报错
        assert!(parse_output_shape(&[2, 12800, 1]).is_err()); // batch != 1
        assert!(parse_output_shape(&[]).is_err());
        assert!(parse_output_shape(&[0, 4]).is_err());
    }

    #[test]
    fn test_group_outputs_six_tensors() {
        // 无 kps 变体: 6 个输出乱序, (N,2) 作 score、(N,4) 作 bbox。
        // 注意 (locations, 2) 宽 score 布局: 640 输入下每 stride 行数是
        // 位置数 (80²/40²/20² = 6400/1600/400), 展平后共 16800 元素。
        let shapes = [
            (800, 4),   // 0: bbox stride 32
            (6400, 2),  // 1: score stride 8 (locations x 2 anchor scores)
            (3200, 4),  // 2: bbox stride 16
            (12800, 4), // 3: bbox stride 8
            (1600, 2),  // 4: score stride 16
            (400, 2),   // 5: score stride 32
        ];
        let g = group_outputs(&shapes).unwrap();

        // 每组按行数降序 → stride 升序 8, 16, 32
        assert_eq!(g.scores, vec![1, 4, 5]);
        assert_eq!(g.bboxes, vec![3, 2, 0]);
        assert!(g.kpss.is_empty());

        // 锚点总数一致性: score 展平总元素 = bbox 总行数 = 16800
        let total_scores: usize = g.scores.iter().map(|&i| shapes[i].0 * shapes[i].1).sum();
        let total_bboxes: usize = g.bboxes.iter().map(|&i| shapes[i].0).sum();
        assert_eq!(total_scores, 16800);
        assert_eq!(total_bboxes, 16800);
    }

    #[test]
    fn test_group_outputs_nine_tensors_rank3() {
        // kps 变体: 9 个 (1, N, C) 形状乱序输出
        let shapes: Vec<(usize, usize)> = vec![
            (3200, 1),   // 0: score stride 16
            (12800, 4),  // 1: bbox stride 8
            (800, 10),   // 2: kps stride 32
            (12800, 1),  // 3: score stride 8
            (3200, 10),  // 4: kps stride 16
            (800, 4),    // 5: bbox stride 32
            (12800, 10), // 6: kps stride 8
            (3200, 4),   // 7: bbox stride 16
            (800, 1),    // 8: score stride 32
        ];
        let g = group_outputs(&shapes).unwrap();

        assert_eq!(g.scores, vec![3, 0, 8]);
        assert_eq!(g.bboxes, vec![1, 7, 5]);
        assert_eq!(g.kpss, vec![6, 4, 2]);
    }

    #[test]
    fn test_group_outputs_rejects_mismatch_and_unknown_dims() {
        // score 与 bbox 数量不一致 → 报错
        let mismatch = [(12800, 1), (3200, 1), (12800, 4)];
        assert!(group_outputs(&mismatch).is_err());

        // 未知维度 → 报错
        let unknown = [(12800, 1), (12800, 4), (12800, 7)];
        assert!(group_outputs(&unknown).is_err());

        // 缺少 bbox → 报错
        let no_bbox = [(12800, 1), (3200, 1)];
        assert!(group_outputs(&no_bbox).is_err());
    }

    #[test]
    fn test_standard_nms_discards_overlaps() {
        // 两个高度重叠的框: 仅保留得分更高者; 独立框保留
        let dets = [
            (0.0, 0.0, 10.0, 10.0, 0.9),   // 0: highest score
            (1.0, 1.0, 11.0, 11.0, 0.8),   // 1: IoU with 0 ≈ 0.68 > 0.4 → 抑制
            (50.0, 50.0, 60.0, 60.0, 0.7), // 2: disjoint → 保留
        ];
        let keep = standard_nms(&dets, 0.4);
        assert_eq!(keep, vec![0, 2]);

        // 阈值边界: IoU 恰等于阈值时不抑制 (ovr <= threshold 保留)
        let boxes = [
            (0.0, 0.0, 4.0, 4.0, 0.9), // 0: area 16
            (2.0, 0.0, 6.0, 4.0, 0.8), // 1: inter 8, union 24, IoU = 1/3
        ];
        let keep = standard_nms(&boxes, 1.0 / 3.0);
        assert_eq!(keep, vec![0, 1]); // IoU == 阈值 → 不丢弃
        let keep = standard_nms(&boxes, 0.3);
        assert_eq!(keep, vec![0]); // IoU > 阈值 → 丢弃

        // 得分降序输出
        let unordered = [
            (0.0, 0.0, 10.0, 10.0, 0.5),
            (100.0, 100.0, 110.0, 110.0, 0.95),
        ];
        assert_eq!(standard_nms(&unordered, 0.4), vec![1, 0]);

        // 空输入
        assert!(standard_nms(&[], 0.4).is_empty());
    }

    #[test]
    fn test_score_layout_flat_indexing_consistency() {
        // (N,1) 与 (N/2,2) 两种 score 布局的展平索引与锚点表一致:
        // 锚点 i ↔ 位置 i/2 的第 i%2 个锚点, (locations,2) 行优先展平后
        // 偏移恰好是 i — 因此统一按 data[i] 读取两种布局。
        let locations = 4usize; // 2x2 特征图, 每位置 2 锚点 → 共 8 锚点
        // (N,1) 密集布局: 每个锚点独立一行
        let dense: Vec<f32> = (0..locations * 2).map(|i| i as f32 + 0.1).collect();
        // (locations, 2) 宽布局: 每位置一行, 两列分别为该位置的锚点 0/1
        let mut wide = vec![0.0f32; locations * 2];
        for loc in 0..locations {
            wide[loc * 2] = (loc * 2) as f32 + 0.1; // 该位置锚点 0
            wide[loc * 2 + 1] = (loc * 2 + 1) as f32 + 0.1; // 锚点 1
        }
        // 统一读取 data[i]: 锚点 i 的分数在两种布局下相同
        for i in 0..locations * 2 {
            assert!(
                (dense[i] - wide[i]).abs() < 1e-6,
                "anchor {} mismatch: dense={} wide={}",
                i,
                dense[i],
                wide[i]
            );
        }
    }

    #[test]
    fn test_preprocess_normalization() {
        // 像素 0 → -1, 255 → +1, 127/128 → ≈0 ((px-127.5)/127.5); 通道序为 BGR
        let mut img = image::RgbImage::new(2, 1);
        img.put_pixel(0, 0, image::Rgb([0u8, 255, 127]));
        img.put_pixel(1, 0, image::Rgb([128u8, 128, 128]));
        let arr = preprocess(&img);
        assert_eq!(arr.shape(), &[1, 3, 1, 2]);
        // (0,0) 像素: R=0,G=255,B=127 → B 通道 (-0.0039), G (+1), R (-1)
        assert!(arr[[0, 0, 0, 0]].abs() < 0.01); // B=127
        assert!((arr[[0, 1, 0, 0]] - 1.0).abs() < 1e-6); // G=255
        assert!((arr[[0, 2, 0, 0]] - (-1.0)).abs() < 1e-6); // R=0
        assert!(arr[[0, 0, 0, 1]].abs() < 0.01); // B=128
    }
}
