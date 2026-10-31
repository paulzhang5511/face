// ====================================================================
// 通用工具函数 — 日志初始化 / IoU / 数值函数
// ====================================================================

/// 初始化日志系统: 安卓平台接入系统日志, 其余平台使用环境变量控制的日志后端。
/// 库本身不会自动调用此函数, 由应用侧在启动时显式调用。
pub fn init_logging() {
    #[cfg(target_os = "android")]
    {
        android_logger::init_once(
            android_logger::Config::default()
                .with_max_level(log::LevelFilter::Debug)
                .with_tag("face-core"),
        );
    }

    #[cfg(not(target_os = "android"))]
    {
        let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
            .format_timestamp_millis()
            .try_init();
    }
}

/// 计算两个边界框的交并比 (IoU), 取值 [0, 1], 用于非极大值抑制
pub fn compute_iou(box_a: (f32, f32, f32, f32), box_b: (f32, f32, f32, f32)) -> f32 {
    // 相交区域: 左上取大, 右下取小
    let x1 = box_a.0.max(box_b.0);
    let y1 = box_a.1.max(box_b.1);
    let x2 = box_a.2.min(box_b.2);
    let y2 = box_a.3.min(box_b.3);

    let inter_w = (x2 - x1).max(0.0);
    let inter_h = (y2 - y1).max(0.0);
    let inter_area = inter_w * inter_h;

    // 并面积 = 两框面积之和 - 交集面积
    let area_a = (box_a.2 - box_a.0) * (box_a.3 - box_a.1);
    let area_b = (box_b.2 - box_b.0) * (box_b.3 - box_b.1);
    let union_area = area_a + area_b - inter_area;

    if union_area <= 0.0 {
        0.0
    } else {
        inter_area / union_area
    }
}

/// Sigmoid 激活函数, 把任意实数压到 (0, 1) 区间
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// 把数值裁剪到 [min, max] 区间
#[inline]
pub fn clamp_f32(val: f32, min: f32, max: f32) -> f32 {
    if val < min {
        min
    } else if val > max {
        max
    } else {
        val
    }
}
