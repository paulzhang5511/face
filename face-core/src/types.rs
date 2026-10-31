// ====================================================================
// face-core: 跨平台人脸美颜引擎
// ====================================================================
// 核心类型定义 — 所有人脸分析结果的统一数据结构

use ndarray::Array2;

/// 边界框 (x1, y1, x2, y2) 像素坐标
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

impl BBox {
    /// 创建新的边界框
    #[inline]
    pub fn new(x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        Self { x1, y1, x2, y2 }
    }

    /// 宽度
    #[inline]
    pub fn width(&self) -> f32 {
        self.x2 - self.x1
    }

    /// 高度
    #[inline]
    pub fn height(&self) -> f32 {
        self.y2 - self.y1
    }

    /// 中心点坐标
    #[inline]
    pub fn center(&self) -> (f32, f32) {
        ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)
    }

    /// 面积
    #[inline]
    pub fn area(&self) -> f32 {
        self.width() * self.height()
    }

    /// 扩展边界框 (用于美颜时扩大裁剪范围以包含头发/耳朵/脖子)
    /// ratios: (left, top, right, bottom), 均为宽/高的比例
    #[must_use]
    pub fn expand(&self, ratios: (f32, f32, f32, f32)) -> Self {
        let w = self.width();
        let h = self.height();
        Self {
            x1: self.x1 - w * ratios.0,
            y1: self.y1 - h * ratios.1,
            x2: self.x2 + w * ratios.2,
            y2: self.y2 + h * ratios.3,
        }
    }

    /// 裁剪边界框到图像范围内
    #[must_use]
    pub fn clamp(&self, img_w: f32, img_h: f32) -> Self {
        Self {
            x1: self.x1.max(0.0).min(img_w),
            y1: self.y1.max(0.0).min(img_h),
            x2: self.x2.max(0.0).min(img_w),
            y2: self.y2.max(0.0).min(img_h),
        }
    }

    /// 转为整数坐标 (用于图像裁剪)
    #[inline]
    pub fn to_int(&self) -> (i32, i32, i32, i32) {
        (
            self.x1 as i32,
            self.y1 as i32,
            self.x2 as i32,
            self.y2 as i32,
        )
    }
}

/// 检测到的人脸 — 包含边界框、置信度、关键点、网格等
#[derive(Debug, Clone)]
pub struct DetectedFace {
    /// 边界框 [x1, y1, x2, y2]
    pub bbox: BBox,
    /// 检测置信度 [0.0, 1.0]
    pub confidence: f32,
    /// 人脸关键点 (K, 2) — 5点/106点/468点 取决于检测器
    pub landmarks: Option<Array2<f32>>,
    /// 3D人脸网格 (468, 3) — FaceMesh结果
    pub mesh_3d: Option<Array2<f32>>,
    /// 网格存在得分 [0.0, 1.0] — FaceMesh返回
    pub mesh_score: Option<f32>,
    /// 追踪ID (多目标追踪)
    pub track_id: Option<i32>,
    /// 人脸识别特征向量 (由识别器填充)
    pub embedding: Option<Vec<f32>>,
    /// 预测性别 (由人口统计模型填充)
    pub gender: Option<Gender>,
    /// 预测年龄, 单位: 年 (由 AgeGender 模型填充)
    pub age: Option<u32>,
    /// 预测年龄段, 如 "20-29" (由 FairFace 模型填充)
    pub age_group: Option<String>,
    /// 预测种族/族裔标签 (由 FairFace 模型填充)
    pub race: Option<String>,
    /// 预测情绪标签 (由情绪模型填充)
    pub emotion: Option<Emotion>,
    /// 情绪预测置信度 [0.0, 1.0]
    pub emotion_confidence: Option<f32>,
    /// 人脸状态五项概率 (由 FaceAttribNet 填充)
    pub face_state: Option<FaceStateResult>,
    /// 人脸图像质量分 [0.0, 1.0] (由 eDifFIQA 填充)
    pub quality: Option<f32>,
    /// 视线估计结果 (由视线模型填充)
    pub gaze: Option<GazeResult>,
    /// 头部姿态结果 (由姿态模型填充)
    pub head_pose: Option<HeadPoseResult>,
    /// 活体检测结果 (由反欺骗模型填充)
    pub spoofing: Option<SpoofingResult>,
}

/// FaceMesh 预测结果 — 468点或478点3D密集面部网格
#[derive(Debug, Clone)]
pub struct FaceMeshResult {
    /// 3D 关键点 (N_points, 3) — x, y 为像素坐标, z 为深度
    pub landmarks_3d: Array2<f32>,
    /// 2D 关键点 (N_points, 2) — 仅 x, y
    pub landmarks_2d: Array2<f32>,
    /// 人脸存在得分 [0.0, 1.0]
    pub score: f32,
}

/// 人脸解析遮罩 — 19类面部区域语义分割结果
#[derive(Debug, Clone)]
pub struct ParsingMask {
    /// 解析遮罩 (H, W) ∈ [0, 18] uint8
    pub mask: Array2<u8>,
    /// 遮罩宽度
    pub width: u32,
    /// 遮罩高度
    pub height: u32,
}

/// 19类面部解析标签定义
pub mod parsing_labels {
    pub const BACKGROUND: u8 = 0;
    pub const SKIN: u8 = 1;
    pub const LEFT_BROW: u8 = 2;
    pub const RIGHT_BROW: u8 = 3;
    pub const LEFT_EYE: u8 = 4;
    pub const RIGHT_EYE: u8 = 5;
    pub const EYE_GLASSES: u8 = 6;
    pub const LEFT_EAR: u8 = 7;
    pub const RIGHT_EAR: u8 = 8;
    pub const EARRING: u8 = 9;
    pub const NOSE: u8 = 10;
    pub const MOUTH: u8 = 11;
    pub const UPPER_LIP: u8 = 12;
    pub const LOWER_LIP: u8 = 13;
    pub const NECK: u8 = 14;
    pub const NECKLACE: u8 = 15;
    pub const CLOTH: u8 = 16;
    pub const HAIR: u8 = 17;
    pub const HAT: u8 = 18;

    /// 所有标签的名称
    pub const LABEL_NAMES: [&str; 19] = [
        "background",
        "skin",
        "l_brow",
        "r_brow",
        "l_eye",
        "r_eye",
        "eye_g",
        "l_ear",
        "r_ear",
        "ear_r",
        "nose",
        "mouth",
        "u_lip",
        "l_lip",
        "neck",
        "neck_l",
        "cloth",
        "hair",
        "hat",
    ];
}

/// 美颜区域类型 — 从解析遮罩中提取的可用区域
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FaceRegion {
    /// 皮肤区域 (磨皮/美白)
    Skin,
    /// 左眉毛
    LeftEyebrow,
    /// 右眉毛
    RightEyebrow,
    /// 左眼
    LeftEye,
    /// 右眼
    RightEye,
    /// 鼻子
    Nose,
    /// 嘴部
    Mouth,
    /// 上唇
    UpperLip,
    /// 下唇
    LowerLip,
}

impl FaceRegion {
    /// 获取对应的解析标签ID
    pub fn label_id(&self) -> u8 {
        match self {
            Self::Skin => parsing_labels::SKIN,
            Self::LeftEyebrow => parsing_labels::LEFT_BROW,
            Self::RightEyebrow => parsing_labels::RIGHT_BROW,
            Self::LeftEye => parsing_labels::LEFT_EYE,
            Self::RightEye => parsing_labels::RIGHT_EYE,
            Self::Nose => parsing_labels::NOSE,
            Self::Mouth => parsing_labels::MOUTH,
            Self::UpperLip => parsing_labels::UPPER_LIP,
            Self::LowerLip => parsing_labels::LOWER_LIP,
        }
    }

    /// 获取区域名称 (英文)
    pub fn name(&self) -> &'static str {
        match self {
            Self::Skin => "skin",
            Self::LeftEyebrow => "left_eyebrow",
            Self::RightEyebrow => "right_eyebrow",
            Self::LeftEye => "left_eye",
            Self::RightEye => "right_eye",
            Self::Nose => "nose",
            Self::Mouth => "mouth",
            Self::UpperLip => "upper_lip",
            Self::LowerLip => "lower_lip",
        }
    }
}

/// 106个面部关键点的索引常量
pub mod landmark_indices {
    // 轮廓 (0-32)
    pub const CONTOUR_START: usize = 0;
    pub const CONTOUR_END: usize = 32;

    // 左眉毛 (33-42)
    pub const LEFT_BROW_START: usize = 33;
    pub const LEFT_BROW_END: usize = 42;

    // 右眉毛 (43-51)
    pub const RIGHT_BROW_START: usize = 43;
    pub const RIGHT_BROW_END: usize = 51;

    // 鼻梁 (52-54)
    pub const NOSE_BRIDGE_START: usize = 52;
    pub const NOSE_BRIDGE_END: usize = 54;

    // 鼻子下部 (55-63)
    pub const NOSE_LOWER_START: usize = 55;
    pub const NOSE_LOWER_END: usize = 63;

    // 左眼 (64-75)
    pub const LEFT_EYE_START: usize = 64;
    pub const LEFT_EYE_END: usize = 75;

    // 右眼 (76-87)
    pub const RIGHT_EYE_START: usize = 76;
    pub const RIGHT_EYE_END: usize = 87;

    // 上唇外侧 (88-94)
    pub const UPPER_LIP_OUTER_START: usize = 88;
    pub const UPPER_LIP_OUTER_END: usize = 94;

    // 下唇外侧 (95-101)
    pub const LOWER_LIP_OUTER_START: usize = 95;
    pub const LOWER_LIP_OUTER_END: usize = 101;

    // 唇部内侧 (102-105)
    pub const LIP_INNER_START: usize = 102;
    pub const LIP_INNER_END: usize = 105;
}

/// 视线估计结果
#[derive(Debug, Clone)]
pub struct GazeResult {
    /// 垂直视线角 (弧度, 正=向上, 负=向下)
    pub pitch: f32,
    /// 水平视线角 (弧度, 正=向右, 负=向左)
    pub yaw: f32,
}

/// 头部姿态估计结果
#[derive(Debug, Clone)]
pub struct HeadPoseResult {
    /// 绕 X 轴旋转 (弧度, 正=低头)
    pub pitch: f32,
    /// 绕 Y 轴旋转 (弧度, 正=向右看)
    pub yaw: f32,
    /// 绕 Z 轴旋转 (弧度, 正=顺时针倾斜)
    pub roll: f32,
}

/// 活体检测结果
#[derive(Debug, Clone)]
pub struct SpoofingResult {
    /// 是否为真实人脸 (true=真人, false=攻击/伪造)
    pub is_real: bool,
    /// 预测置信度 [0.0, 1.0]
    pub confidence: f32,
}

/// 人脸图像质量评估结果
#[derive(Debug, Clone)]
pub struct QualityResult {
    /// 质量分 [0.0, 1.0], 越高越好
    pub score: f32,
}

/// 情绪类别 (对齐情绪模型标签顺序)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Emotion {
    Neutral,
    Happy,
    Sad,
    Surprise,
    Fear,
    Disgust,
    Anger,
    Contempt,
}

impl Emotion {
    /// 获取情绪标签字符串
    pub fn label(&self) -> &'static str {
        match self {
            Self::Neutral => "Neutral",
            Self::Happy => "Happy",
            Self::Sad => "Sad",
            Self::Surprise => "Surprise",
            Self::Fear => "Fear",
            Self::Disgust => "Disgust",
            Self::Anger => "Angry",
            Self::Contempt => "Contempt",
        }
    }
}

/// 情绪识别结果
#[derive(Debug, Clone)]
pub struct EmotionResult {
    /// 预测的情绪类别
    pub emotion: Emotion,
    /// 预测置信度 [0.0, 1.0]
    pub confidence: f32,
}

/// 性别
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gender {
    Female,
    Male,
}

/// 人口统计属性预测结果 (AgeGender / FairFace 统一结构)
#[derive(Debug, Clone)]
pub struct DemographyResult {
    /// 预测性别
    pub gender: Gender,
    /// 精确年龄 (年), 仅 AgeGender 模型提供
    pub age: Option<u32>,
    /// 年龄段字符串, 如 "20-29", 仅 FairFace 模型提供
    pub age_group: Option<String>,
    /// 种族/族裔标签, 仅 FairFace 模型提供
    pub race: Option<String>,
}

/// 人脸状态预测结果 (FaceAttribNet) — 五项独立的二分类概率
///
/// 各值来自独立分类头, 不求和为1, 可同时为高 (例如同时戴墨镜和口罩); 需分别按阈值判定
#[derive(Debug, Clone)]
pub struct FaceStateResult {
    /// 左眼睁开的概率
    pub left_eye_open: f32,
    /// 右眼睁开的概率
    pub right_eye_open: f32,
    /// 佩戴眼镜的概率
    pub eyeglasses: f32,
    /// 佩戴口罩的概率
    pub mask: f32,
    /// 佩戴墨镜的概率
    pub sunglasses: f32,
}

/// 人脸抠图 (matting) 结果 — alpha 遮罩
#[derive(Debug, Clone)]
pub struct MatteResult {
    /// Alpha 遮罩 (H, W) ∈ [0.0, 1.0]
    pub alpha: Array2<f32>,
    /// 遮罩宽度
    pub width: u32,
    /// 遮罩高度
    pub height: u32,
}
