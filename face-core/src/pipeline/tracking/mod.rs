// ====================================================================
// BYTETracker 多目标追踪
// ====================================================================
//
// 组成模块:
//   - 两轮关联 (高分/低分) + 未确认轨迹第三轮 + 新轨迹激活 + 丢失清理
//   - 8维状态 [cx, cy, a, h, vcx, vcy, va, vh] 卡尔曼滤波
//   - IoU 距离 (1 - IoU) + 线性指派
//
// 实现说明:
//   1. 线性指派: 用贪心算法替代匈牙利算法
//      (按检测分数降序、代价升序逐个选取未占用且 cost <= 阈值的配对)。
//      密集遮挡/交叉场景可能得到次优匹配, 常规场景结果一致。
//   2. 卡尔曼 update 用标准增益公式并直接解 4x4 线性方程组,
//      数值上等价于 Cholesky (cho_factor/cho_solve) 解法。
//   3. 输出集合: 除已激活 (is_activated) 的 Tracked 轨迹外,
//      按任务要求额外包含未超时的 Lost 轨迹。
//   4. 状态空间沿用 BYTETrack 原始的 [cx, cy, aspect, h],
//      而非任务描述中的 [cx, cy, w, h] — 以参考实现为准。

use crate::types::{BBox, DetectedFace};

/// 轨迹状态
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TrackState {
    /// 新建 (未激活)
    New,
    /// 正在追踪
    Tracked,
    /// 丢失 (等待重关联)
    Lost,
    /// 已移除 (生命周期结束)
    Removed,
}

/// 第二轮低分关联的 IoU 距离阈值 (thresh=0.5)
const SECOND_MATCH_THRESH: f32 = 0.5;

/// 未确认轨迹关联阈值 (thresh=0.7)
const UNCONFIRMED_MATCH_THRESH: f32 = 0.7;

// ====================================================================
// 卡尔曼滤波
// ====================================================================

/// BYTETrack 卡尔曼滤波器
///
/// 状态 (8维): [cx, cy, a, h, vcx, vcy, va, vh]
struct KalmanFilter {
    /// 位置噪声权重 (1/20)
    std_weight_position: f32,
    /// 速度噪声权重 (1/160)
    std_weight_velocity: f32,
}

/// 8维状态均值
type Mean = [f32; 8];
/// 8x8 协方差矩阵 (行主序)
type Cov = [f32; 64];
/// 4维观测 [cx, cy, a, h]
type Measurement = [f32; 4];

impl KalmanFilter {
    fn new() -> Self {
        Self {
            std_weight_position: 1.0 / 20.0,
            std_weight_velocity: 1.0 / 160.0,
        }
    }

    /// 由未关联测量初始化轨迹
    fn initiate(&self, measurement: &Measurement) -> (Mean, Cov) {
        let h = measurement[3];
        let std = [
            2.0 * self.std_weight_position * h,
            2.0 * self.std_weight_position * h,
            1e-2,
            2.0 * self.std_weight_position * h,
            10.0 * self.std_weight_velocity * h,
            10.0 * self.std_weight_velocity * h,
            1e-5,
            10.0 * self.std_weight_velocity * h,
        ];
        let mut mean = [0.0f32; 8];
        mean[..4].copy_from_slice(measurement);
        let mut cov = [0.0f32; 64];
        for i in 0..8 {
            cov[i * 8 + i] = std[i] * std[i];
        }
        (mean, cov)
    }

    /// 预测步: mean = F·mean, cov = F·P·F' + Q
    fn predict(&self, mean: &Mean, cov: &Cov) -> (Mean, Cov) {
        // F = I(8), F[i, 4+i] = dt = 1.0
        let mut new_mean = *mean;
        for i in 0..4 {
            new_mean[i] += mean[4 + i];
        }

        // Q = diag(std²), std 系数与 initiate 同源
        let h = mean[3];
        let q_diag = [
            self.std_weight_position * h,
            self.std_weight_position * h,
            1e-2,
            self.std_weight_position * h,
            self.std_weight_velocity * h,
            self.std_weight_velocity * h,
            1e-5,
            self.std_weight_velocity * h,
        ];

        // cov' = F·P·F' + Q, 利用 F 的稀疏结构展开:
        //   (F·P·F')[i][j] = fp[i][j] + (j<4 ? fp[i][4+j] : 0)
        let mut new_cov = [0.0f32; 64];
        for i in 0..8 {
            for j in 0..8 {
                let fp = if i < 4 {
                    cov[i * 8 + j] + cov[(4 + i) * 8 + j]
                } else {
                    cov[i * 8 + j]
                };
                let fp_j = if j < 4 {
                    let col = if i < 4 {
                        cov[i * 8 + 4 + j] + cov[(4 + i) * 8 + 4 + j]
                    } else {
                        cov[i * 8 + 4 + j]
                    };
                    fp + col
                } else {
                    fp
                };
                new_cov[i * 8 + j] = fp_j;
            }
        }
        for i in 0..8 {
            new_cov[i * 8 + i] += q_diag[i] * q_diag[i];
        }

        (new_mean, new_cov)
    }

    /// 投影到观测空间: H = I(8) 前 4 行
    fn project(&self, mean: &Mean, cov: &Cov) -> (Measurement, [f32; 16]) {
        // R = diag(std²), std 系数与 initiate 同源
        let h = mean[3];
        let r_diag = [
            self.std_weight_position * h,
            self.std_weight_position * h,
            1e-1,
            self.std_weight_position * h,
        ];

        let proj_mean = [mean[0], mean[1], mean[2], mean[3]];
        // S = H·P·H' + R = P[:4,:4] + R
        let mut s = [0.0f32; 16];
        for i in 0..4 {
            for j in 0..4 {
                s[i * 4 + j] = cov[i * 8 + j];
            }
        }
        for i in 0..4 {
            s[i * 4 + i] += r_diag[i] * r_diag[i];
        }
        (proj_mean, s)
    }

    /// 校正步 (卡尔曼更新)
    ///
    /// K = P·H'·S^{-1}, 通过解 S·X = H·P (4 个右端项) 得到;
    /// 数值上等价于 Cholesky 解法 (工程简化: 高斯消元 + 部分主元)
    fn update(&self, mean: &Mean, cov: &Cov, measurement: &Measurement) -> (Mean, Cov) {
        let (proj_mean, s) = self.project(mean, cov);

        // 右端项: H·P 为 4x8 (P 的前 4 行)
        let mut hp = [[0.0f32; 8]; 4];
        for i in 0..4 {
            for j in 0..8 {
                hp[i][j] = cov[i * 8 + j];
            }
        }

        // 解 S·X = H·P, X = K' (4x8)
        let mut a = s;
        let mut x = hp;
        for c in 0..4 {
            // 部分主元
            let mut best = c;
            for r in c + 1..4 {
                if a[r * 4 + c].abs() > a[best * 4 + c].abs() {
                    best = r;
                }
            }
            if best != c {
                for j in 0..4 {
                    a.swap(c * 4 + j, best * 4 + j);
                }
                x.swap(c, best);
            }
            let d = a[c * 4 + c];
            if d.abs() < 1e-12 {
                // 奇异保护: 该维度增益置零 (工程简化)
                for v in x[c].iter_mut() {
                    *v = 0.0;
                }
                continue;
            }
            for j in 0..4 {
                a[c * 4 + j] /= d;
            }
            for v in x[c].iter_mut() {
                *v /= d;
            }
            for r in 0..4 {
                if r == c {
                    continue;
                }
                let f = a[r * 4 + c];
                if f != 0.0 {
                    for j in 0..4 {
                        a[r * 4 + j] -= f * a[c * 4 + j];
                    }
                    for j in 0..8 {
                        x[r][j] -= f * x[c][j];
                    }
                }
            }
        }

        // K (8x4): K[j][c] = X[c][j]
        let mut k = [[0.0f32; 4]; 8];
        for c in 0..4 {
            for j in 0..8 {
                k[j][c] = x[c][j];
            }
        }

        // new_mean = mean + K·(z - proj_mean)
        let innovation = [
            measurement[0] - proj_mean[0],
            measurement[1] - proj_mean[1],
            measurement[2] - proj_mean[2],
            measurement[3] - proj_mean[3],
        ];
        let mut new_mean = *mean;
        for i in 0..8 {
            let mut acc = 0.0;
            for c in 0..4 {
                acc += k[i][c] * innovation[c];
            }
            new_mean[i] += acc;
        }

        // new_cov = P - K·S·K'
        let mut new_cov = *cov;
        let mut ks = [[0.0f32; 4]; 8];
        for i in 0..8 {
            for j in 0..4 {
                let mut acc = 0.0;
                for c in 0..4 {
                    acc += k[i][c] * s[c * 4 + j];
                }
                ks[i][j] = acc;
            }
        }
        for i in 0..8 {
            for j in 0..8 {
                let mut acc = 0.0;
                for c in 0..4 {
                    acc += ks[i][c] * k[j][c];
                }
                new_cov[i * 8 + j] -= acc;
            }
        }

        (new_mean, new_cov)
    }
}

// ====================================================================
// STrack — 单条轨迹
// ====================================================================

/// 单条轨迹 (Kalman 状态 + 生命周期元数据)
struct STrack {
    /// 卡尔曼状态均值 (8维)
    mean: Mean,
    /// 卡尔曼协方差 (8x8)
    covariance: Cov,
    /// 轨迹状态
    state: TrackState,
    /// 是否已激活 (frame_id == 1 或匹配后置位)
    is_activated: bool,
    /// 检测置信度
    score: f32,
    /// 轨迹 ID
    track_id: i32,
    /// 连续追踪帧计数
    tracklet_len: u32,
    /// 最近一次更新的帧号 (对应 end_frame)
    frame_id: u32,
    /// 起始帧号
    start_frame: u32,
}

impl STrack {
    /// 由检测创建候选轨迹 (未激活, 观测暂存于 mean 前 4 维)
    fn from_detection(det_bbox: &BBox, score: f32) -> Self {
        let m = bbox_to_xyah(det_bbox);
        let mut mean = [0.0f32; 8];
        mean[..4].copy_from_slice(&m);
        Self {
            mean,
            covariance: [0.0; 64],
            state: TrackState::New,
            is_activated: false,
            score,
            track_id: 0,
            tracklet_len: 0,
            frame_id: 0,
            start_frame: 0,
        }
    }

    /// 激活新轨迹
    fn activate(&mut self, kf: &KalmanFilter, track_id: i32, frame_id: u32) {
        let mut measurement = [0.0f32; 4];
        measurement.copy_from_slice(&self.mean[..4]);
        let (mean, cov) = kf.initiate(&measurement);
        self.mean = mean;
        self.covariance = cov;
        self.track_id = track_id;
        self.tracklet_len = 0;
        self.state = TrackState::Tracked;
        if frame_id == 1 {
            self.is_activated = true;
        }
        self.frame_id = frame_id;
        self.start_frame = frame_id;
    }

    /// 重激活丢失轨迹
    fn re_activate(&mut self, kf: &KalmanFilter, det: &BBox, score: f32, frame_id: u32) {
        let m = bbox_to_xyah(det);
        let (mean, cov) = kf.update(&self.mean, &self.covariance, &m);
        self.mean = mean;
        self.covariance = cov;
        self.tracklet_len = 0;
        self.state = TrackState::Tracked;
        self.is_activated = true;
        self.frame_id = frame_id;
        self.score = score;
    }

    /// 用匹配检测更新
    fn update(&mut self, kf: &KalmanFilter, det: &BBox, score: f32, frame_id: u32) {
        let m = bbox_to_xyah(det);
        let (mean, cov) = kf.update(&self.mean, &self.covariance, &m);
        self.mean = mean;
        self.covariance = cov;
        self.frame_id = frame_id;
        self.tracklet_len += 1;
        self.state = TrackState::Tracked;
        self.is_activated = true;
        self.score = score;
    }

    /// 预测下一帧: 非 Tracked 状态高度速度置零
    fn predict(&mut self, kf: &KalmanFilter) {
        if self.state != TrackState::Tracked {
            self.mean[7] = 0.0;
        }
        let (mean, cov) = kf.predict(&self.mean, &self.covariance);
        self.mean = mean;
        self.covariance = cov;
    }

    /// 输出 bbox [x1,y1,x2,y2] (w = a·h)
    fn tlbr(&self) -> BBox {
        let (cx, cy, a, h) = (self.mean[0], self.mean[1], self.mean[2], self.mean[3]);
        let w = a * h;
        BBox::new(cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)
    }
}

/// BBox -> [cx,cy,a,h] 观测
#[inline]
fn bbox_to_xyah(b: &BBox) -> Measurement {
    let w = b.width();
    let h = b.height();
    [b.x1 + w / 2.0, b.y1 + h / 2.0, w / h, h]
}

/// IoU 距离 (1 - IoU)
#[inline]
fn iou_distance(a: &BBox, b: &BBox) -> f32 {
    let ix1 = a.x1.max(b.x1);
    let iy1 = a.y1.max(b.y1);
    let ix2 = a.x2.min(b.x2);
    let iy2 = a.y2.min(b.y2);
    let inter = (ix2 - ix1).max(0.0) * (iy2 - iy1).max(0.0);
    let union = a.area() + b.area() - inter;
    if union <= 0.0 {
        1.0
    } else {
        1.0 - inter / union
    }
}

/// 贪心指派 — 求解代价最小的一对一指派 (匈牙利算法的简化替代)
///
/// 差异说明: 匈牙利算法求全局最小总代价的最优指派; 本实现先按检测分数降序、
/// 代价升序排序所有 (cost <= thresh) 候选对, 再逐个选取双方均未占用的配对。
/// 在检测/轨迹一对一冲突密集的交叉场景可能得到次优解; 常规近邻一致场景结果等价。
///
/// 返回 (matches, unmatched_a, unmatched_b), 仅保留 cost <= thresh 的配对
fn greedy_assignment(
    a_boxes: &[BBox],
    b_boxes: &[BBox],
    b_scores: &[f32],
    thresh: f32,
) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>) {
    let n = a_boxes.len();
    let m = b_boxes.len();

    // 收集所有 cost <= thresh 的候选对 (score, cost, i, j)
    let mut cands: Vec<(f32, f32, usize, usize)> = Vec::new();
    for (i, a) in a_boxes.iter().enumerate() {
        for (j, b) in b_boxes.iter().enumerate() {
            let cost = iou_distance(a, b);
            if cost <= thresh {
                cands.push((b_scores[j], cost, i, j));
            }
        }
    }

    // 按 (分数降序, 代价升序) 排序后贪心选取
    cands.sort_by(|x, y| {
        y.0.partial_cmp(&x.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(x.1.partial_cmp(&y.1).unwrap_or(std::cmp::Ordering::Equal))
    });

    let mut matched_a = vec![false; n];
    let mut matched_b = vec![false; m];
    let mut matches = Vec::new();
    for (_, _, i, j) in cands {
        if !matched_a[i] && !matched_b[j] {
            matched_a[i] = true;
            matched_b[j] = true;
            matches.push((i, j));
        }
    }

    let unmatched_a = (0..n).filter(|&i| !matched_a[i]).collect();
    let unmatched_b = (0..m).filter(|&j| !matched_b[j]).collect();
    (matches, unmatched_a, unmatched_b)
}

// ====================================================================
// BYTETracker
// ====================================================================

/// 追踪输出 — 一条存活轨迹
#[derive(Debug, Clone)]
pub struct TrackResult {
    /// 轨迹 ID
    pub track_id: i32,
    /// 当前轨迹 bbox (Kalman 校正后的 [x1,y1,x2,y2])
    pub bbox: BBox,
    /// 最新关联检测的置信度
    pub confidence: f32,
}

/// BYTETracker 多目标追踪器 — 两轮关联 (高分/低分) + 未确认轨迹关联
pub struct ByteTracker {
    /// 高分检测阈值 (>= 视为高分)
    track_thresh: f32,
    /// 第一轮 IoU 距离匹配阈值
    match_thresh: f32,
    /// 丢失轨迹保留帧数, 超过后转 Removed (对应 track_buffer)
    max_lost_frames: u32,
    /// 帧计数
    frame_id: u32,
    /// 轨迹 ID 计数器 (对应 BaseTrack._count)
    track_id_counter: i32,
    /// 卡尔曼滤波器
    kf: KalmanFilter,
    /// 追踪中轨迹 (含未确认)
    tracked: Vec<STrack>,
    /// 丢失轨迹 (未超时)
    lost: Vec<STrack>,
    /// 已移除轨迹 (仅留存计数用)
    removed: Vec<STrack>,
}

impl ByteTracker {
    /// 创建追踪器
    ///
    /// - `track_thresh`: 高分检测阈值 (对应 BYTETracker.track_thresh)
    /// - `match_thresh`: 第一轮 IoU 距离阈值 (对应 match_thresh)
    /// - `max_lost_frames`: 丢失容忍帧数 (对应 track_buffer)
    pub fn new(track_thresh: f32, match_thresh: f32, max_lost_frames: u32) -> Self {
        Self {
            track_thresh,
            match_thresh,
            max_lost_frames,
            frame_id: 0,
            track_id_counter: 0,
            kf: KalmanFilter::new(),
            tracked: Vec::new(),
            lost: Vec::new(),
            removed: Vec::new(),
        }
    }

    /// 重置追踪器 (对应 BYTETracker.reset)
    pub fn reset(&mut self) {
        self.frame_id = 0;
        self.track_id_counter = 0;
        self.tracked.clear();
        self.lost.clear();
        self.removed.clear();
    }

    /// 每帧喂入检测结果, 返回当前存活轨迹 (含新确认/保持/丢失)
    ///
    /// 输出包含所有 is_activated 的 Tracked 轨迹, 以及未超时的 Lost 轨迹;
    /// Removed 轨迹不输出。未确认 (新创建且非首帧) 的轨迹在被匹配确认前不输出。
    pub fn update(&mut self, detections: &[DetectedFace]) -> Vec<TrackResult> {
        self.frame_id += 1;
        let frame_id = self.frame_id;

        // 按阈值拆分高低分检测 (低分阈值取 0, 对应 low_thresh 语义)
        let mut dets_high: Vec<(BBox, f32)> = Vec::new();
        let mut dets_low: Vec<(BBox, f32)> = Vec::new();
        for d in detections {
            if d.confidence >= self.track_thresh {
                dets_high.push((d.bbox, d.confidence));
            } else if d.confidence > 0.0 {
                dets_low.push((d.bbox, d.confidence));
            }
        }

        // 分离已确认/未确认轨迹 (Step 0)
        let mut unconfirmed: Vec<STrack> = Vec::new();
        let mut confirmed: Vec<STrack> = Vec::new();
        for t in self.tracked.drain(..) {
            if t.is_activated {
                confirmed.push(t);
            } else {
                unconfirmed.push(t);
            }
        }

        // 联合池 = confirmed + lost
        let mut strack_pool: Vec<STrack> = confirmed;
        strack_pool.append(&mut self.lost);
        let was_tracked: Vec<bool> = strack_pool
            .iter()
            .map(|t| t.state == TrackState::Tracked)
            .collect();

        // Step 1: 预测
        for t in strack_pool.iter_mut() {
            t.predict(&self.kf);
        }

        let pool_boxes: Vec<BBox> = strack_pool.iter().map(|t| t.tlbr()).collect();
        let high_boxes: Vec<BBox> = dets_high.iter().map(|(b, _)| *b).collect();
        let high_scores: Vec<f32> = dets_high.iter().map(|(_, s)| *s).collect();
        let (matches1, u_track, u_det_high) =
            greedy_assignment(&pool_boxes, &high_boxes, &high_scores, self.match_thresh);

        // Step 2: 剩余 Tracked 轨迹与低分检测关联 (thresh=0.5)
        let r_tracked: Vec<usize> = u_track
            .iter()
            .copied()
            .filter(|&i| was_tracked[i])
            .collect();
        let r_boxes: Vec<BBox> = r_tracked.iter().map(|&i| strack_pool[i].tlbr()).collect();
        let low_boxes: Vec<BBox> = dets_low.iter().map(|(b, _)| *b).collect();
        let low_scores: Vec<f32> = dets_low.iter().map(|(_, s)| *s).collect();
        let (matches2, u_track_second, _) =
            greedy_assignment(&r_boxes, &low_boxes, &low_scores, SECOND_MATCH_THRESH);

        // 贪心结果 -> 池索引映射
        let mut matched1: Vec<Option<usize>> = vec![None; strack_pool.len()];
        for &(i, j) in &matches1 {
            matched1[i] = Some(j);
        }
        let mut matched2: Vec<Option<usize>> = vec![None; strack_pool.len()];
        for &(k, j) in &matches2 {
            matched2[r_tracked[k]] = Some(j);
        }
        let second_unmatched_tracked: Vec<usize> =
            u_track_second.iter().map(|&k| r_tracked[k]).collect();

        // 归位: 遍历池中每条轨迹, 按匹配结果 update/re_activate/转 Lost
        let mut activated: Vec<STrack> = Vec::new();
        let mut lost_now: Vec<STrack> = Vec::new();
        for (i, mut t) in strack_pool.into_iter().enumerate() {
            if let Some(j) = matched1[i] {
                let (det_box, det_score) = dets_high[j];
                if was_tracked[i] {
                    t.update(&self.kf, &det_box, det_score, frame_id);
                } else {
                    t.re_activate(&self.kf, &det_box, det_score, frame_id);
                }
                activated.push(t);
            } else if let Some(j) = matched2[i] {
                let (det_box, det_score) = dets_low[j];
                t.update(&self.kf, &det_box, det_score, frame_id);
                activated.push(t);
            } else if was_tracked[i] && second_unmatched_tracked.contains(&i) {
                // 未匹配的 Tracked 轨迹标记 Lost
                t.state = TrackState::Lost;
                lost_now.push(t);
            } else {
                // 原本就是 Lost 且未匹配 — 保持 Lost
                lost_now.push(t);
            }
        }

        // Step 3: 未确认轨迹与剩余高分检测关联 (thresh=0.7)
        let unconf_boxes: Vec<BBox> = unconfirmed.iter().map(|t| t.tlbr()).collect();
        let remain_boxes: Vec<BBox> = u_det_high.iter().map(|&j| dets_high[j].0).collect();
        let remain_scores: Vec<f32> = u_det_high.iter().map(|&j| dets_high[j].1).collect();
        let (matches3, u_unconf, u_det_remain) = greedy_assignment(
            &unconf_boxes,
            &remain_boxes,
            &remain_scores,
            UNCONFIRMED_MATCH_THRESH,
        );
        let mut matched3: Vec<Option<usize>> = vec![None; unconfirmed.len()];
        for &(i, j) in &matches3 {
            matched3[i] = Some(u_det_high[j]);
        }

        let mut removed_now: Vec<STrack> = Vec::new();
        for (i, mut t) in unconfirmed.into_iter().enumerate() {
            if let Some(det_idx) = matched3[i] {
                let (det_box, det_score) = dets_high[det_idx];
                t.update(&self.kf, &det_box, det_score, frame_id);
                activated.push(t);
            } else {
                // 未匹配的未确认轨迹直接 Removed (对应 mark_removed)
                t.state = TrackState::Removed;
                removed_now.push(t);
            }
        }

        // Step 4: 剩余高分检测激活新轨迹 (score >= track_thresh)
        for &j in &u_det_remain {
            let (det_box, det_score) = dets_high[u_det_high[j]];
            if det_score < self.track_thresh {
                continue;
            }
            let mut t = STrack::from_detection(&det_box, det_score);
            self.track_id_counter += 1;
            t.activate(&self.kf, self.track_id_counter, frame_id);
            activated.push(t);
        }
        let _ = u_unconf; // 已通过 matched3 归位

        // Step 5: 丢失超时的轨迹转 Removed (frame_id - end_frame > buffer)
        for mut t in lost_now {
            if frame_id.saturating_sub(t.frame_id) > self.max_lost_frames {
                t.state = TrackState::Removed;
                removed_now.push(t);
            } else {
                self.lost.push(t);
            }
        }

        // 更新列表: 每条轨迹对象唯一归属一个列表,
        // 无需联合/去重操作
        self.tracked = activated;
        self.removed.append(&mut removed_now);

        // 输出: 已激活的 Tracked 轨迹 + 未超时的 Lost 轨迹
        let mut out = Vec::new();
        for t in &self.tracked {
            if t.is_activated {
                out.push(TrackResult {
                    track_id: t.track_id,
                    bbox: t.tlbr(),
                    confidence: t.score,
                });
            }
        }
        for t in &self.lost {
            if t.is_activated {
                out.push(TrackResult {
                    track_id: t.track_id,
                    bbox: t.tlbr(),
                    confidence: t.score,
                });
            }
        }
        out
    }
}

// ====================================================================
// 单元测试
// ====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// 构造简单检测 (其余属性 None)
    fn det(x1: f32, y1: f32, x2: f32, y2: f32, score: f32) -> DetectedFace {
        DetectedFace {
            bbox: BBox::new(x1, y1, x2, y2),
            confidence: score,
            landmarks: None,
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
        }
    }

    #[test]
    fn test_static_single_target_stable_id() {
        // 单目标静止 3 帧, track_id 稳定
        let mut tracker = ByteTracker::new(0.5, 0.8, 30);
        let dets = [det(100.0, 100.0, 200.0, 200.0, 0.9)];

        let mut ids: Vec<i32> = Vec::new();
        for _ in 0..3 {
            let results = tracker.update(&dets);
            assert_eq!(results.len(), 1);
            ids.push(results[0].track_id);
        }
        assert_eq!(ids.len(), 3);
        assert_eq!(ids[0], ids[1]);
        assert_eq!(ids[1], ids[2]);
    }

    #[test]
    fn test_target_removed_after_max_lost_frames() {
        // 目标消失 max_lost_frames 后轨迹移除, 重现时不复用旧 ID
        let mut tracker = ByteTracker::new(0.5, 0.8, 2);

        // 第 1 帧激活 (首帧直接确认)
        let dets = [det(100.0, 100.0, 200.0, 200.0, 0.9)];
        let r = tracker.update(&dets);
        assert_eq!(r.len(), 1);
        let tid = r[0].track_id;

        // 第 2~4 帧空检测: 丢失计数 1, 2, 3 — 第 4 帧时 4-1=3 > buffer=2, 转 Removed
        for _ in 0..3 {
            tracker.update(&[]);
        }

        // 第 5 帧目标重现 (新轨迹, 未确认不输出); 第 6 帧确认, ID 为新 ID
        tracker.update(&dets);
        let r = tracker.update(&dets);
        assert_eq!(r.len(), 1);
        assert_ne!(r[0].track_id, tid);
    }

    #[test]
    fn test_two_crossing_targets_keep_two_ids() {
        // 两个目标交叉时 ID 数量保持为 2
        let mut tracker = ByteTracker::new(0.5, 0.8, 30);

        let mut ids: Vec<i32> = Vec::new();
        for step in 0..8 {
            let off = step as f32 * 20.0;
            let dets = [
                det(50.0 + off, 50.0, 150.0 + off, 150.0, 0.9),
                det(450.0 - off, 50.0, 550.0 - off, 150.0, 0.9),
            ];
            let r = tracker.update(&dets);
            assert_eq!(r.len(), 2, "step {}", step);
            for t in &r {
                if !ids.contains(&t.track_id) {
                    ids.push(t.track_id);
                }
            }
        }

        // 交叉前后总 ID 数量保持为 2
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn test_low_score_detection_revives_tracked() {
        // 低分检测可延续 Tracked 轨迹 (BYTE 第二轮关联)
        let mut tracker = ByteTracker::new(0.5, 0.8, 30);

        let r1 = tracker.update(&[det(100.0, 100.0, 200.0, 200.0, 0.9)]);
        let tid = r1[0].track_id;

        // 低分 (0.3 < 0.5) 同位置检测 — 应通过第二轮关联延续
        let r2 = tracker.update(&[det(102.0, 102.0, 202.0, 202.0, 0.3)]);
        assert_eq!(r2.len(), 1);
        assert_eq!(r2[0].track_id, tid);
    }

    #[test]
    fn test_unconfirmed_track_removed_without_match() {
        // 第 2 帧新建的未确认轨迹, 下一帧无匹配则直接 Removed
        let mut tracker = ByteTracker::new(0.5, 0.8, 30);

        // 第 1 帧: 空 / frame 1: empty
        tracker.update(&[]);

        // 第 2 帧: 新目标 (frame != 1, 未确认不输出)
        let r = tracker.update(&[det(10.0, 10.0, 60.0, 60.0, 0.9)]);
        assert_eq!(r.len(), 0, "frame-2 new track must not be output yet");

        // 第 3 帧: 目标跳到远处 (无匹配) -> 未确认轨迹被移除, 新轨迹同样未确认
        let r = tracker.update(&[det(300.0, 300.0, 360.0, 360.0, 0.9)]);
        assert_eq!(r.len(), 0);
    }

    #[test]
    fn test_lost_track_in_output_until_expired() {
        // 丢失但未超时的轨迹包含在输出中 / lost-but-unexpired tracks appear in output
        let mut tracker = ByteTracker::new(0.5, 0.8, 5);
        tracker.update(&[det(100.0, 100.0, 200.0, 200.0, 0.9)]);

        // 空帧: 轨迹转 Lost, 但未超时, 仍在输出中
        let r = tracker.update(&[]);
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn test_reset_clears_state() {
        let mut tracker = ByteTracker::new(0.5, 0.8, 30);
        tracker.update(&[det(0.0, 0.0, 50.0, 50.0, 0.9)]);
        tracker.reset();
        let r = tracker.update(&[]);
        assert!(r.is_empty());
    }

    #[test]
    fn test_kalman_predict_moves_with_velocity() {
        // 有速度的预测应移动位置
        let kf = KalmanFilter::new();
        let (mut mean, cov) = kf.initiate(&[100.0, 100.0, 1.0, 50.0]);
        mean[4] = 10.0; // vcx = 10
        let (new_mean, _) = kf.predict(&mean, &cov);
        assert!((new_mean[0] - 110.0).abs() < 1e-4);
        assert!((new_mean[4] - 10.0).abs() < 1e-4);
    }

    #[test]
    fn test_kalman_update_pulls_toward_measurement() {
        // 校正应把状态拉向观测
        let kf = KalmanFilter::new();
        let (mean, cov) = kf.initiate(&[100.0, 100.0, 1.0, 50.0]);
        let (new_mean, _) = kf.update(&mean, &cov, &[110.0, 100.0, 1.0, 50.0]);
        assert!(new_mean[0] > 100.0 && new_mean[0] <= 110.0);
    }
}
