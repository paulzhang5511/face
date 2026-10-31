// ====================================================================
// face-core: 跨平台人脸美颜引擎 — 库入口
// ====================================================================
//
// 模块架构:
//   types      — 统一数据类型 (BBox, DetectedFace, ParsingMask等)
//   models     — 模型字节内嵌管理
//   inference  — ONNX Runtime推理引擎封装
//   pipeline   — 美颜管线:
//       detector  — BlazeFace人脸检测
//       facemesh  — FaceMesh 468点3D网格
//       bisenet   — BiSeNet 19类语义分割
//   utils      — 工具:
//       image_utils — 图像处理 (缩放/裁剪/归一化)
//       alignment   — 人脸对齐 (仿射变换)
//       common      — 通用工具 (IoU/NMS/sigmoid/日志)

pub mod error;
pub mod inference;
pub mod models;
pub mod pipeline;
pub mod types;
pub mod utils;

/// 预导入 — 库使用者常用项
pub mod prelude {
    pub use crate::error::{FaceError, Result};
    pub use crate::inference::{DetectorKind, EngineConfig};
    pub use crate::types::{BBox, DetectedFace, FaceRegion, MatteResult, ParsingMask};
    pub use crate::{AnalysisOptions, StageTimings};
    pub use crate::{AnalyzedFace, FaceAnalyzer, FaceEngine, FaceResult};
}

use crate::error::{FaceError, Result};
use crate::types::DetectedFace;
use image::{DynamicImage, GenericImageView};
use log::{info, warn};

use crate::inference::{DetectorKind, EngineConfig, InferenceSession};

/// 美颜引擎 — 统一的人脸分析接口
///
/// 整合了人脸检测 + 3D网格 + 语义分割的完整管线
///
/// # 使用示例
///
/// ```ignore
/// use face_core::FaceEngine;
/// use face_core::inference::EngineConfig;
///
/// let config = EngineConfig::default();
/// let mut engine = FaceEngine::new(config).unwrap();
/// let image = engine.load_image("face.jpg").unwrap();
/// let results = engine.analyze(&image).unwrap();
/// for face in results {
///     println!("Face at {:?}, confidence={:.2}", face.face.bbox, face.face.confidence);
///     if let Some(mask) = &face.parsing_mask {
///         println!("  Parsing mask: {}x{}", mask.width, mask.height);
///     }
/// }
/// ```
pub struct FaceEngine {
    /// 引擎配置
    #[allow(dead_code)]
    config: EngineConfig,
    /// 人脸检测器 (BlazeFace)
    detector: Option<pipeline::detector::BlazeFaceDetector>,
    /// 人脸网格 (FaceMesh 468点)
    facemesh: Option<pipeline::facemesh::FaceMesh>,
    /// 语义分割 (BiSeNet 19类)
    bisenet: Option<pipeline::bisenet::BiSeNet>,
}

/// 单个人脸的美颜分析结果
#[derive(Debug, Clone)]
pub struct AnalyzedFace {
    /// 检测到的人脸基本信息
    pub face: crate::types::DetectedFace,
    /// 3D 密集人脸网格 (468点)
    pub mesh_result: Option<crate::types::FaceMeshResult>,
    /// 面部语义分割遮罩 (19类)
    pub parsing_mask: Option<crate::types::ParsingMask>,
}

/// 美颜分析的完整结果
#[derive(Debug, Clone)]
pub struct FaceResult {
    /// 检测到的所有人脸分析结果
    pub faces: Vec<AnalyzedFace>,
    /// 原始图像宽度
    pub image_width: u32,
    /// 原始图像高度
    pub image_height: u32,
}

impl FaceEngine {
    /// 创建美颜引擎 (从配置自动加载所有内嵌模型)
    ///
    /// # 参数
    /// * `config` - 引擎配置 (指定启用哪些功能)
    ///
    /// # 返回
    /// 初始化好的美颜引擎
    pub fn new(config: EngineConfig) -> Result<Self> {
        info!("FaceEngine: initializing...");
        crate::models::log_embedded_models();

        // 1. 加载人脸检测器
        let detector = if config.enable_detection {
            let model_bytes = crate::models::get_model_bytes("blazeface_128")
                .ok_or_else(|| FaceError::Other("BlazeFace model not embedded".into()))?;
            let session = InferenceSession::from_bytes("blazeface_128", model_bytes)?;
            let detector = pipeline::detector::BlazeFaceDetector::from_session(
                session,
                config.detection_threshold,
                0.3, // 加权NMS IoU阈值
            )?;
            info!("FaceEngine: face detector loaded");
            Some(detector)
        } else {
            None
        };

        // 2. 加载人脸网格
        let facemesh = if config.enable_facemesh {
            let model_bytes = crate::models::get_model_bytes("facemesh_468")
                .ok_or_else(|| FaceError::Other("FaceMesh model not embedded".into()))?;
            let session = InferenceSession::from_bytes("facemesh_468", model_bytes)?;
            let facemesh = pipeline::facemesh::FaceMesh::from_session(session, 0.25)?;
            info!("FaceEngine: face mesh (468pts) loaded");
            Some(facemesh)
        } else {
            None
        };

        // 3. 加载语义分割
        let bisenet = if config.enable_parsing {
            let model_bytes = crate::models::get_model_bytes("bisenet_r18")
                .ok_or_else(|| FaceError::Other("BiSeNet model not embedded".into()))?;
            let session = InferenceSession::from_bytes("bisenet_r18", model_bytes)?;
            let bisenet = pipeline::bisenet::BiSeNet::from_session(session)?;
            info!("FaceEngine: face parsing (BiSeNet 19 classes) loaded");
            Some(bisenet)
        } else {
            None
        };

        info!("FaceEngine: initialization complete");
        Ok(Self {
            config,
            detector,
            facemesh,
            bisenet,
        })
    }

    /// 从文件加载图像
    pub fn load_image(&self, path: &str) -> Result<DynamicImage> {
        // 按 IMREAD_COLOR 语义: 应用 EXIF 方向
        let img = crate::utils::image_utils::load_image_with_exif(path)?;
        info!(
            "FaceEngine: loaded image {}x{} from {}",
            img.width(),
            img.height(),
            path
        );
        Ok(img)
    }

    /// 从字节加载图像 (Android Camera2/NV21 数据需要先转换)
    pub fn load_image_from_bytes(&self, bytes: &[u8]) -> Result<DynamicImage> {
        Ok(crate::utils::image_utils::load_image_from_bytes(bytes)?)
    }

    /// 对图像执行完整美颜分析
    ///
    /// 管线顺序: 检测 -> FaceMesh (如有) -> 语义分割 (如有)
    ///
    /// # 参数
    /// * `image` - 输入图像 (RGB/BGR)
    ///
    /// # 返回
    /// 包含所有人脸完整分析结果的 FaceResult
    pub fn analyze(&mut self, image: &DynamicImage) -> Result<FaceResult> {
        let (img_w, img_h) = image.dimensions();
        info!("FaceEngine: analyzing {}x{} image", img_w, img_h);

        // 第1步: 人脸检测
        let detected_faces = match &mut self.detector {
            Some(det) => det.detect(image)?,
            None => {
                info!("FaceEngine: detection disabled, returning empty result");
                return Ok(FaceResult {
                    faces: Vec::new(),
                    image_width: img_w,
                    image_height: img_h,
                });
            }
        };

        if detected_faces.is_empty() {
            info!("FaceEngine: no faces detected");
            return Ok(FaceResult {
                faces: Vec::new(),
                image_width: img_w,
                image_height: img_h,
            });
        }

        info!("FaceEngine: detected {} face(s)", detected_faces.len());

        // 第2步 + 第3步: 对每张人脸执行FaceMesh和语义分割
        let mut beauty_faces: Vec<AnalyzedFace> = Vec::with_capacity(detected_faces.len());

        for face in detected_faces {
            // FaceMesh 3D网格
            let mesh_result = match &mut self.facemesh {
                Some(fm) => fm.predict(image, &face).ok(),
                None => None,
            };

            // BiSeNet 语义分割
            let parsing_mask = match &mut self.bisenet {
                Some(bn) => bn.parse(image, &face).ok(),
                None => None,
            };

            beauty_faces.push(AnalyzedFace {
                face,
                mesh_result,
                parsing_mask,
            });
        }

        info!(
            "FaceEngine: analysis complete — {} faces, {} with mesh, {} with parsing",
            beauty_faces.len(),
            beauty_faces
                .iter()
                .filter(|f| f.mesh_result.is_some())
                .count(),
            beauty_faces
                .iter()
                .filter(|f| f.parsing_mask.is_some())
                .count(),
        );

        Ok(FaceResult {
            faces: beauty_faces,
            image_width: img_w,
            image_height: img_h,
        })
    }

    /// 仅执行人脸检测 (快速模式)
    pub fn detect_only(&mut self, image: &DynamicImage) -> Result<Vec<crate::types::DetectedFace>> {
        match &mut self.detector {
            Some(det) => Ok(det.detect(image)?),
            None => Ok(Vec::new()),
        }
    }
}

// ====================================================================
// FaceAnalyzer — 统一人脸分析编排
// ====================================================================
//
// 编排语义:
//   1. detect → 每个脸逐项 enrich (predictor 失败仅 log warn 不中断)
//   2. 识别 / 质量只在检测器产出 5 点对齐 landmarks 时可用
//      (supports_alignment: SCRFD use_kps 有; BlazeFace 6 点无, warn-skip)
//   3. FaceMesh 结果用 estimate_from_mesh 计算几何头姿写回 face.head_pose
//   4. matting 对整图独立运行 (matting() 方法)

/// 统一人脸分析器 — 检测器 + 识别器 + 预测器组合
///
/// # 使用示例
///
/// ```ignore
/// use face_core::FaceAnalyzer;
/// use face_core::inference::{DetectorKind, EngineConfig};
///
/// let config = EngineConfig {
///     detector_kind: DetectorKind::Scrfd,
///     enable_recognition: true,
///     enable_attributes: true,
///     ..Default::default()
/// };
/// let mut analyzer = FaceAnalyzer::new(config).unwrap();
/// let faces = analyzer.analyze(&image).unwrap();
/// ```
/// 分阶段耗时统计 (毫秒), 由 analyze_profiled 填充
///
/// batch predictor 记录一次批量调用的总耗时;
/// 逐脸 predictor (emotion/gaze/headpose/spoofing) 累加全部人脸的单次耗时
#[derive(Debug, Default, Clone)]
pub struct StageTimings {
    /// 人脸检测 (SCRFD/BlazeFace)
    pub detect: f64,
    /// FaceMesh 468 点网格
    pub facemesh: f64,
    /// ArcFace 特征嵌入
    pub recognition: f64,
    /// 年龄/性别
    pub age_gender: f64,
    /// 情绪 (FER+)
    pub emotion: f64,
    /// 人脸状态 (FaceAttribNet)
    pub face_attrib: f64,
    /// 视线 (MobileNetV2)
    pub gaze: f64,
    /// 头部姿态 (ResNet18)
    pub headpose: f64,
    /// 质量评估 (eDifFIQA)
    pub quality: f64,
    /// 活体检测 (MiniFASNet)
    pub spoofing: f64,
}

impl StageTimings {
    /// 全部阶段耗时之和
    pub fn total(&self) -> f64 {
        self.detect
            + self.facemesh
            + self.recognition
            + self.age_gender
            + self.emotion
            + self.face_attrib
            + self.gaze
            + self.headpose
            + self.quality
            + self.spoofing
    }

    /// 逐阶段 (名称, 耗时) 迭代
    pub fn iter(&self) -> impl Iterator<Item = (&'static str, f64)> {
        [
            ("detect", self.detect),
            ("facemesh", self.facemesh),
            ("recognition", self.recognition),
            ("age_gender", self.age_gender),
            ("emotion", self.emotion),
            ("face_attrib", self.face_attrib),
            ("gaze", self.gaze),
            ("headpose", self.headpose),
            ("quality", self.quality),
            ("spoofing", self.spoofing),
        ]
        .into_iter()
    }
}

pub struct FaceAnalyzer {
    /// 引擎配置副本 (max_faces 等)
    config: EngineConfig,
    /// 检测器: BlazeFace 或 SCRFD
    detector: Detector,
    /// 人脸网格 (FaceMesh 468点), 供几何头姿等使用
    facemesh: Option<pipeline::facemesh::FaceMesh>,
    /// 识别器 (MBF 架构, 512维嵌入)
    recognizer: Option<pipeline::recognition::ArcFace>,
    /// 年龄/性别预测器
    age_gender: Option<pipeline::attribute::AgeGender>,
    /// 情绪预测器 (AffectNet8)
    emotion: Option<pipeline::attribute::EmotionNet>,
    /// 人脸状态预测器 (FaceAttribNet)
    face_attrib: Option<pipeline::attribute::FaceAttribNet>,
    /// 视线估计器 (MobileNetV2 Gaze)
    gaze: Option<pipeline::gaze::GazeNet>,
    /// 头部姿态模型 (ResNet18)
    headpose_net: Option<pipeline::headpose::HeadPoseNet>,
    /// 质量评估器 (eDifFIQA)
    quality: Option<pipeline::quality::EDifFIQA>,
    /// 活体检测器 (MiniFASNet)
    spoofing: Option<pipeline::spoofing::MiniFASNet>,
    /// 人像抠图 (MODNet), 整图独立运行
    modnet: Option<pipeline::matting::ModNet>,
    /// 多目标追踪器 (BYTETracker)
    tracker: Option<pipeline::tracking::ByteTracker>,
}

/// 内部检测器枚举 — BlazeFace 或 SCRFD
enum Detector {
    /// BlazeFace 检测器 (6 关键点, 不支持对齐)
    Blaze(pipeline::detector::BlazeFaceDetector),
    /// SCRFD 检测器 (5 点对齐关键点)
    Scrfd(pipeline::detector_scrfd::ScrfdDetector),
}

impl Detector {
    /// 运行检测
    fn detect(&mut self, image: &DynamicImage) -> Result<Vec<crate::types::DetectedFace>> {
        match self {
            Self::Blaze(det) => Ok(det.detect(image)?),
            Self::Scrfd(det) => Ok(det.detect(image)?),
        }
    }

    /// 是否产出 5 点对齐 landmarks (supports_alignment 语义)
    fn supports_alignment(&self) -> bool {
        matches!(self, Self::Scrfd(_))
    }
}

/// 分析选项 — 覆盖 max_faces 等 (分析时行为开关)
#[derive(Debug, Clone, Default)]
pub struct AnalysisOptions {
    /// 本次分析最大人脸数 (0 = 不限制; 覆盖 EngineConfig.max_faces)
    pub max_faces: usize,
}

impl FaceAnalyzer {
    /// 当前平台推理线程数 (与 InferenceSession::from_bytes 内部策略一致)
    fn num_cpu_threads() -> usize {
        #[cfg(target_os = "android")]
        {
            4
        }
        #[cfg(not(target_os = "android"))]
        {
            num_cpus::get().max(1)
        }
    }

    /// 创建推理会话
    fn make_session(name: &str) -> Result<InferenceSession> {
        let bytes = crate::models::get_model_bytes(name)
            .ok_or_else(|| FaceError::Other(format!("{} model not embedded", name)))?;
        let threads = match name {
            "emotion_ferplus" => 2,
            _ => Self::num_cpu_threads(),
        };
        Ok(InferenceSession::from_bytes_with_threads(
            name, bytes, threads,
        )?)
    }

    /// 创建统一人脸分析器 (从配置自动加载所有启用的模型)
    ///
    /// # 参数
    /// * `config` - 引擎配置 (指定检测器类型与各功能开关)
    pub fn new(config: EngineConfig) -> Result<Self> {
        info!("FaceAnalyzer: initializing...");
        crate::models::log_embedded_models();

        // 1. 检测器 (BlazeFace 或 SCRFD)
        let detector = if config.enable_detection {
            match config.detector_kind {
                DetectorKind::BlazeFace => {
                    let session = Self::make_session("blazeface_128")?;
                    Detector::Blaze(pipeline::detector::BlazeFaceDetector::from_session(
                        session,
                        config.detection_threshold,
                        0.3, // 加权NMS IoU阈值
                    )?)
                }
                DetectorKind::Scrfd => {
                    let session = Self::make_session("scrfd_500m_kps")?;
                    // use_kps=true: 产出 5 点对齐 landmarks (supports_alignment)
                    Detector::Scrfd(pipeline::detector_scrfd::ScrfdDetector::from_session(
                        session,
                        config.detection_threshold,
                        0.4, // 标准 NMS IoU 阈值
                        true,
                    )?)
                }
            }
        } else {
            return Err(FaceError::NotEnabled(
                "FaceAnalyzer requires a detector; enable_detection must be true".into(),
            ));
        };

        // 2. FaceMesh (供几何头姿 / 后续美颜使用)
        let facemesh = if config.enable_facemesh {
            let session = Self::make_session("facemesh_468")?;
            info!("FaceAnalyzer: face mesh (468pts) loaded");
            Some(pipeline::facemesh::FaceMesh::from_session(session, 0.25)?)
        } else {
            None
        };

        // 3. 识别器 (仅当检测器支持 5 点对齐; 构造期检查对齐能力)
        let recognizer = if config.enable_recognition {
            if detector.supports_alignment() {
                let session = Self::make_session("arcface_mbf")?;
                info!("FaceAnalyzer: recognition enabled (ArcFace MBF)");
                Some(pipeline::recognition::ArcFace::from_session(session)?)
            } else {
                // 不可对齐时禁用识别并仅告警
                warn!(
                    "FaceAnalyzer: detector does not produce alignment landmarks; recognition disabled"
                );
                None
            }
        } else {
            None
        };

        // 4. 属性四合一: 年龄/性别 + 情绪 + 人脸状态
        let (age_gender, emotion, face_attrib) = if config.enable_attributes {
            let ag =
                pipeline::attribute::AgeGender::from_session(Self::make_session("genderage")?)?;
            let em = pipeline::attribute::EmotionNet::from_session(
                Self::make_session("emotion_ferplus")?,
                pipeline::attribute::EmotionModel::FerPlus, // FER+ ONNX (AffectNet 系列仅 TorchScript, ORT 无法加载)
            )?;
            let fa = pipeline::attribute::FaceAttribNet::from_session(
                Self::make_session("face_attrib_net")?,
                0.0, // margin 使用默认值 0.0
            )?;
            info!("FaceAnalyzer: attributes enabled (AgeGender + Emotion + FaceAttribNet)");
            (Some(ag), Some(em), Some(fa))
        } else {
            (None, None, None)
        };

        // 5. 视线估计
        let gaze = if config.enable_gaze {
            let session = Self::make_session("mobilenetv2_gaze")?;
            info!("FaceAnalyzer: gaze enabled (MobileNetV2)");
            Some(pipeline::gaze::GazeNet::from_session(session)?)
        } else {
            None
        };

        // 6. 头部姿态模型 (ResNet18; FaceMesh 几何头姿作为无模型时的回退)
        let headpose_net = if config.enable_headpose {
            let session = Self::make_session("resnet18_headpose")?;
            info!("FaceAnalyzer: head pose enabled (ResNet18)");
            Some(pipeline::headpose::HeadPoseNet::from_session(session)?)
        } else {
            None
        };

        // 7. 质量评估 (仅当检测器支持 5 点对齐)
        let quality = if config.enable_quality {
            if detector.supports_alignment() {
                let session = Self::make_session("ediffiqa")?;
                info!("FaceAnalyzer: quality enabled (eDifFIQA)");
                Some(pipeline::quality::EDifFIQA::from_session(session)?)
            } else {
                warn!(
                    "FaceAnalyzer: detector does not produce alignment landmarks; quality disabled"
                );
                None
            }
        } else {
            None
        };

        // 8. 活体检测
        let spoofing = if config.enable_spoofing {
            let session = Self::make_session("minifasnet")?;
            info!("FaceAnalyzer: anti-spoofing enabled (MiniFASNet)");
            Some(pipeline::spoofing::MiniFASNet::from_session(session)?)
        } else {
            None
        };

        // 9. 人像抠图 (整图独立运行, 由 matting() 方法调用)
        let modnet = if config.enable_matting {
            let session = Self::make_session("modnet")?;
            info!("FaceAnalyzer: matting enabled (MODNet)");
            Some(pipeline::matting::ModNet::from_session(session)?)
        } else {
            None
        };

        // 10. 追踪器
        let tracker = if config.enable_tracking {
            info!("FaceAnalyzer: tracking enabled (BYTETracker)");
            Some(pipeline::tracking::ByteTracker::new(
                config.detection_threshold,
                0.8, // 匹配阈值 match_thresh=0.8
                30,  // 缓冲帧数 track_buffer=30
            ))
        } else {
            None
        };

        info!("FaceAnalyzer: initialization complete");
        Ok(Self {
            config,
            detector,
            facemesh,
            recognizer,
            age_gender,
            emotion,
            face_attrib,
            gaze,
            headpose_net,
            quality,
            spoofing,
            modnet,
            tracker,
        })
    }

    /// 对图像执行统一分析
    ///
    /// 流程: 检测 → 逐脸 enrich (每项 predictor 失败仅 log warn 不中断)
    ///
    /// 识别 / 质量仅在检测器产出 5 点对齐 landmarks 时运行
    /// (BlazeFace 下 warn-skip, 对齐 supports_alignment 语义)。
    pub fn analyze(&mut self, image: &DynamicImage) -> Result<Vec<crate::types::DetectedFace>> {
        let faces = self.analyze_with_options(image, &AnalysisOptions::default())?;
        Ok(faces)
    }

    /// 分阶段计时的统一分析 (供性能测试工具使用)
    ///
    /// 返回各阶段累计耗时 (毫秒), 阶段语义与 analyze_with_options 一致;
    /// batch predictor (识别/质量/年龄性别/人脸状态/FaceMesh) 按"一次批量调用"计为单阶段耗时
    pub fn analyze_profiled(
        &mut self,
        image: &DynamicImage,
    ) -> Result<(Vec<crate::types::DetectedFace>, StageTimings)> {
        let mut t = StageTimings::default();
        let t0 = std::time::Instant::now();
        let mut faces = match self.detector.detect(image) {
            Ok(f) => f,
            Err(e) => {
                warn!(
                    "FaceAnalyzer: face detection failed, returning empty result: {}",
                    e
                );
                Vec::new()
            }
        };
        t.detect = t0.elapsed().as_secs_f64() * 1000.0;

        let options = AnalysisOptions::default();
        let max_faces = if options.max_faces > 0 {
            options.max_faces
        } else {
            self.config.max_faces
        };
        if max_faces > 0 && faces.len() > max_faces {
            faces.sort_by(|a, b| {
                b.confidence
                    .partial_cmp(&a.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            faces.truncate(max_faces);
        }

        let single_face = faces.len() <= 1;

        // 3a. 识别 (batch)
        if let Some(rec) = &mut self.recognizer {
            let indices: Vec<usize> = faces
                .iter()
                .enumerate()
                .filter(|(_, f)| f.landmarks.is_some())
                .map(|(i, _)| i)
                .collect();
            let t0 = std::time::Instant::now();
            let batch_out = {
                let refs: Vec<&DetectedFace> = indices.iter().map(|&i| &faces[i]).collect();
                match rec.get_normalized_embeddings_batch(image, &refs) {
                    Ok(embeds) => Some(embeds),
                    Err(e) => {
                        warn!(
                            "FaceAnalyzer: batch embedding failed, falling back to per-face: {}",
                            e
                        );
                        None
                    }
                }
            };
            let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
            match batch_out {
                Some(embeds) => {
                    for (k, emb) in embeds.into_iter().enumerate() {
                        if let Some(e) = emb {
                            let idx = indices[k];
                            faces[idx].embedding = Some(e);
                        }
                    }
                }
                None => {
                    for (_idx, face) in faces.iter_mut().enumerate() {
                        if let Some(landmarks) = &face.landmarks {
                            if let Ok(embedding) = rec.get_normalized_embedding(image, landmarks) {
                                face.embedding = Some(embedding);
                            }
                        }
                    }
                }
            }
            t.recognition = elapsed;
        }

        // 3b. FaceMesh (batch)
        let mut mesh_results: Vec<Option<crate::types::FaceMeshResult>> = vec![None; faces.len()];
        if let Some(fm) = &mut self.facemesh {
            let t0 = std::time::Instant::now();
            let batch_out = if !single_face {
                match fm.predict_batch(image, &faces) {
                    Ok(results) => Some(results),
                    Err(e) => {
                        warn!(
                            "FaceAnalyzer: batch FaceMesh failed, falling back to per-face: {}",
                            e
                        );
                        None
                    }
                }
            } else {
                None
            };
            match batch_out {
                Some(results) => {
                    let mut it = results.into_iter();
                    for (idx, face) in faces.iter_mut().enumerate() {
                        if let Some(mr) = it.next() {
                            face.mesh_3d = Some(mr.landmarks_3d.clone());
                            face.mesh_score = Some(mr.score);
                            mesh_results[idx] = Some(mr);
                        }
                    }
                }
                None => {
                    for (idx, face) in faces.iter_mut().enumerate() {
                        if let Ok(mr) = fm.predict(image, face) {
                            face.mesh_3d = Some(mr.landmarks_3d.clone());
                            face.mesh_score = Some(mr.score);
                            mesh_results[idx] = Some(mr);
                        }
                    }
                }
            }
            t.facemesh = t0.elapsed().as_secs_f64() * 1000.0;
        }

        // 3c'. 年龄/性别 (batch, 多脸)
        if !single_face {
            if let Some(ag) = &mut self.age_gender {
                let t0 = std::time::Instant::now();
                let refs: Vec<&DetectedFace> = faces.iter().collect();
                let out = match ag.predict_batch(image, &refs) {
                    Ok(results) => {
                        for (i, r) in results.into_iter().enumerate() {
                            if let Some(demo) = r {
                                faces[i].gender = Some(demo.gender);
                                faces[i].age = demo.age;
                            }
                        }
                        true
                    }
                    Err(_) => false,
                };
                if !out {
                    for (_idx, face) in faces.iter_mut().enumerate() {
                        if let Ok(demo) = ag.predict(image, face) {
                            face.gender = Some(demo.gender);
                            face.age = demo.age;
                        }
                    }
                }
                t.age_gender = t0.elapsed().as_secs_f64() * 1000.0;
            }
        }

        // 3e'. 人脸状态 (batch, 多脸)
        if !single_face {
            if let Some(fa) = &mut self.face_attrib {
                let t0 = std::time::Instant::now();
                let refs: Vec<&DetectedFace> = faces.iter().collect();
                let ok = match fa.predict_batch(image, &refs) {
                    Ok(results) => {
                        for (i, r) in results.into_iter().enumerate() {
                            if let Some(fs) = r {
                                faces[i].face_state = Some(fs);
                            }
                        }
                        true
                    }
                    Err(_) => false,
                };
                if !ok {
                    for (_idx, face) in faces.iter_mut().enumerate() {
                        if let Ok(fs) = fa.predict(image, face) {
                            face.face_state = Some(fs);
                        }
                    }
                }
                t.face_attrib = t0.elapsed().as_secs_f64() * 1000.0;
            }
        }

        // 3h'. 质量 (batch)
        if let Some(q) = &mut self.quality {
            let indices: Vec<usize> = faces
                .iter()
                .enumerate()
                .filter(|(_, f)| f.landmarks.is_some())
                .map(|(i, _)| i)
                .collect();
            if !indices.is_empty() {
                let t0 = std::time::Instant::now();
                let refs: Vec<&DetectedFace> = indices.iter().map(|&i| &faces[i]).collect();
                let out = match q.predict_batch(image, &refs) {
                    Ok(results) => {
                        for (k, r) in results.into_iter().enumerate() {
                            if let Some(qr) = r {
                                let idx = indices[k];
                                faces[idx].quality = Some(qr.score);
                            }
                        }
                        true
                    }
                    Err(_) => false,
                };
                if !out {
                    for (_idx, face) in faces.iter_mut().enumerate() {
                        if let Some(landmarks) = &face.landmarks {
                            if let Ok(qr) = q.predict(image, landmarks) {
                                face.quality = Some(qr.score);
                            }
                        }
                    }
                }
                t.quality = t0.elapsed().as_secs_f64() * 1000.0;
            }
        }

        // 3c-3i: 逐脸 (age_gender/face_attrib 单脸 + emotion/gaze/headpose/spoofing)
        let (img_w, img_h) = image.dimensions();
        for (_idx, face) in faces.iter_mut().enumerate() {
            if let Some(ag) = &mut self.age_gender {
                if single_face {
                    let t0 = std::time::Instant::now();
                    if let Ok(demo) = ag.predict(image, face) {
                        face.gender = Some(demo.gender);
                        face.age = demo.age;
                    }
                    t.age_gender += t0.elapsed().as_secs_f64() * 1000.0;
                }
            }
        }

        // 3d'. 情绪批量 (多脸)
        if !single_face {
            if let Some(em) = &mut self.emotion {
                let t0 = std::time::Instant::now();
                let refs: Vec<&DetectedFace> = faces.iter().collect();
                let ok = match em.predict_batch(image, &refs) {
                    Ok(results) => {
                        for (i, r) in results.into_iter().enumerate() {
                            if let Some(er) = r {
                                faces[i].emotion = Some(er.emotion);
                                faces[i].emotion_confidence = Some(er.confidence);
                            }
                        }
                        true
                    }
                    Err(_) => false,
                };
                if !ok {
                    for (_idx, face) in faces.iter_mut().enumerate() {
                        if let Ok(er) = em.predict(image, face) {
                            face.emotion = Some(er.emotion);
                            face.emotion_confidence = Some(er.confidence);
                        }
                    }
                }
                t.emotion = t0.elapsed().as_secs_f64() * 1000.0;
            }
        }

        // 3f'. 视线批量 (多脸)
        if !single_face {
            if let Some(gz) = &mut self.gaze {
                let t0 = std::time::Instant::now();
                let refs: Vec<&DetectedFace> = faces.iter().collect();
                let ok = match gz.predict_batch(image, &refs) {
                    Ok(results) => {
                        for (i, r) in results.into_iter().enumerate() {
                            if let Some(g) = r {
                                faces[i].gaze = Some(g);
                            }
                        }
                        true
                    }
                    Err(_) => false,
                };
                if !ok {
                    for (_idx, face) in faces.iter_mut().enumerate() {
                        if let Ok(g) = gz.predict(image, face) {
                            face.gaze = Some(g);
                        }
                    }
                }
                t.gaze = t0.elapsed().as_secs_f64() * 1000.0;
            }
        }

        // 3g'. 头部姿态批量 (多脸)
        if !single_face && self.config.enable_headpose {
            if let Some(hp) = &mut self.headpose_net {
                let t0 = std::time::Instant::now();
                let refs: Vec<&DetectedFace> = faces.iter().collect();
                let ok = match hp.predict_batch(image, &refs) {
                    Ok(results) => {
                        for (i, r) in results.into_iter().enumerate() {
                            if let Some(pose) = r {
                                faces[i].head_pose = Some(pose);
                            }
                        }
                        true
                    }
                    Err(_) => false,
                };
                if !ok {
                    for (_idx, face) in faces.iter_mut().enumerate() {
                        if let Ok(r) = hp.predict(image, face) {
                            face.head_pose = Some(r);
                        }
                    }
                }
                t.headpose = t0.elapsed().as_secs_f64() * 1000.0;
            }
        }

        for (idx, face) in faces.iter_mut().enumerate() {
            if self.config.enable_headpose && face.head_pose.is_none() {
                if let Some(mr) = &mesh_results[idx] {
                    let pose =
                        pipeline::headpose::estimate_from_mesh(&mr.landmarks_3d, img_w, img_h);
                    face.head_pose = Some(pose);
                }
            }
        }

        for (_idx, face) in faces.iter_mut().enumerate() {
            if let Some(em) = &mut self.emotion {
                if single_face {
                    let t0 = std::time::Instant::now();
                    if let Ok(er) = em.predict(image, face) {
                        face.emotion = Some(er.emotion);
                        face.emotion_confidence = Some(er.confidence);
                    }
                    t.emotion += t0.elapsed().as_secs_f64() * 1000.0;
                }
            }

            if let Some(gz) = &mut self.gaze {
                if single_face {
                    let t0 = std::time::Instant::now();
                    if let Ok(g) = gz.predict(image, face) {
                        face.gaze = Some(g);
                    }
                    t.gaze += t0.elapsed().as_secs_f64() * 1000.0;
                }
            }

            if let Some(hp) = &mut self.headpose_net {
                if single_face && self.config.enable_headpose {
                    let t0 = std::time::Instant::now();
                    if let Ok(r) = hp.predict(image, face) {
                        face.head_pose = Some(r);
                    }
                    t.headpose += t0.elapsed().as_secs_f64() * 1000.0;
                }
            }

            if let Some(sp) = &mut self.spoofing {
                let t0 = std::time::Instant::now();
                if let Ok(sr) = sp.predict(image, &face.bbox) {
                    face.spoofing = Some(sr);
                }
                t.spoofing += t0.elapsed().as_secs_f64() * 1000.0;
            }
        }

        Ok((faces, t))
    }

    /// 对图像执行统一分析, 附带每次调用的选项覆盖
    pub fn analyze_with_options(
        &mut self,
        image: &DynamicImage,
        options: &AnalysisOptions,
    ) -> Result<Vec<crate::types::DetectedFace>> {
        // 第1步: 人脸检测 (失败仅 log warn 不中断, 逐项 predictor 语义)
        let mut faces = match self.detector.detect(image) {
            Ok(f) => f,
            Err(e) => {
                warn!(
                    "FaceAnalyzer: face detection failed, returning empty result: {}",
                    e
                );
                Vec::new()
            }
        };
        info!("FaceAnalyzer: detected {} face(s)", faces.len());

        // 第2步: max_faces 截断 (置信度降序)
        let max_faces = if options.max_faces > 0 {
            options.max_faces
        } else {
            self.config.max_faces
        };
        if max_faces > 0 && faces.len() > max_faces {
            faces.sort_by(|a, b| {
                b.confidence
                    .partial_cmp(&a.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            faces.truncate(max_faces);
            info!(
                "FaceAnalyzer: truncated to {} face(s) by max_faces",
                max_faces
            );
        }

        // 第3步: enrich — 支持 batch 的 predictor 一次推理处理全部人脸,
        // 不支持 batch 的 (facemesh/emotion/gaze/headpose/spoofing) 逐脸执行
        let (img_w, img_h) = image.dimensions();
        let supports_alignment = self.detector.supports_alignment();

        // 3a. 识别: 批量对齐 + 单次 batch 推理 (仅 5 点对齐)
        if let Some(rec) = &mut self.recognizer {
            let indices: Vec<usize> = faces
                .iter()
                .enumerate()
                .filter(|(_, f)| f.landmarks.is_some())
                .map(|(i, _)| i)
                .collect();
            let batch_out = {
                let refs: Vec<&DetectedFace> = indices.iter().map(|&i| &faces[i]).collect();
                match rec.get_normalized_embeddings_batch(image, &refs) {
                    Ok(embeds) => Some(embeds),
                    Err(e) => {
                        warn!(
                            "FaceAnalyzer: batch embedding failed, falling back to per-face: {}",
                            e
                        );
                        None
                    }
                }
            };
            match batch_out {
                Some(embeds) => {
                    for (k, emb) in embeds.into_iter().enumerate() {
                        if let Some(e) = emb {
                            let idx = indices[k];
                            faces[idx].embedding = Some(e.clone());
                            info!(
                                "FaceAnalyzer: face {} embedding extracted ({}-D)",
                                idx + 1,
                                e.len()
                            );
                        }
                    }
                }
                None => {
                    for (idx, face) in faces.iter_mut().enumerate() {
                        if let Some(landmarks) = &face.landmarks {
                            match rec.get_normalized_embedding(image, landmarks) {
                                Ok(embedding) => face.embedding = Some(embedding),
                                Err(e2) => warn!(
                                    "FaceAnalyzer: face {} failed to extract embedding: {}",
                                    idx + 1,
                                    e2
                                ),
                            }
                        }
                    }
                }
            }
        }

        // 3b. FaceMesh: 批量推理 (多脸), 单脸走逐脸 predict
        let single_face = faces.len() <= 1;
        let mut mesh_results: Vec<Option<crate::types::FaceMeshResult>> = vec![None; faces.len()];
        if let Some(fm) = &mut self.facemesh {
            let batch_out = if !single_face {
                match fm.predict_batch(image, &faces) {
                    Ok(results) => Some(results),
                    Err(e) => {
                        warn!(
                            "FaceAnalyzer: batch FaceMesh failed, falling back to per-face: {}",
                            e
                        );
                        None
                    }
                }
            } else {
                None
            };
            match batch_out {
                Some(results) => {
                    // predict_batch 跳过预处理失败的脸上, 按序对齐到成功预测的脸
                    let mut it = results.into_iter();
                    for (idx, face) in faces.iter_mut().enumerate() {
                        if let Some(mr) = it.next() {
                            face.mesh_3d = Some(mr.landmarks_3d.clone());
                            face.mesh_score = Some(mr.score);
                            mesh_results[idx] = Some(mr);
                        }
                    }
                }
                None => {
                    for (idx, face) in faces.iter_mut().enumerate() {
                        match fm.predict(image, face) {
                            Ok(mr) => {
                                face.mesh_3d = Some(mr.landmarks_3d.clone());
                                face.mesh_score = Some(mr.score);
                                mesh_results[idx] = Some(mr);
                            }
                            Err(e) => warn!(
                                "FaceAnalyzer: face {} FaceMesh prediction failed: {}",
                                idx + 1,
                                e
                            ),
                        }
                    }
                }
            }
        }

        // 3c-3i: 逐脸执行不支持 batch 的 predictor
        for (idx, face) in faces.iter_mut().enumerate() {
            // 3c. 属性: 年龄/性别 — 单脸时直接 predict, 多脸走 batch (见 3c')
            if let Some(ag) = &mut self.age_gender {
                if single_face {
                    match ag.predict(image, face) {
                        Ok(demo) => {
                            face.gender = Some(demo.gender);
                            face.age = demo.age;
                        }
                        Err(e) => warn!(
                            "FaceAnalyzer: face {} AgeGender prediction failed: {}",
                            idx + 1,
                            e
                        ),
                    }
                }
            }

            // 3i. 活体检测
            if let Some(sp) = &mut self.spoofing {
                match sp.predict(image, &face.bbox) {
                    Ok(sr) => face.spoofing = Some(sr),
                    Err(e) => warn!("FaceAnalyzer: face {} anti-spoofing failed: {}", idx + 1, e),
                }
            }
        }

        // 3d'. 情绪批量 (多脸)
        if !single_face {
            if let Some(em) = &mut self.emotion {
                let refs: Vec<&DetectedFace> = faces.iter().collect();
                match em.predict_batch(image, &refs) {
                    Ok(results) => {
                        for (i, r) in results.into_iter().enumerate() {
                            if let Some(er) = r {
                                faces[i].emotion = Some(er.emotion);
                                faces[i].emotion_confidence = Some(er.confidence);
                            }
                        }
                    }
                    Err(e) => {
                        warn!(
                            "FaceAnalyzer: batch emotion failed, falling back to per-face: {}",
                            e
                        );
                        for (idx, face) in faces.iter_mut().enumerate() {
                            match em.predict(image, face) {
                                Ok(er) => {
                                    face.emotion = Some(er.emotion);
                                    face.emotion_confidence = Some(er.confidence);
                                }
                                Err(e2) => warn!(
                                    "FaceAnalyzer: face {} emotion prediction failed: {}",
                                    idx + 1,
                                    e2
                                ),
                            }
                        }
                    }
                }
            }
        }

        // 3f'. 视线批量 (多脸)
        if !single_face {
            if let Some(gz) = &mut self.gaze {
                let refs: Vec<&DetectedFace> = faces.iter().collect();
                match gz.predict_batch(image, &refs) {
                    Ok(results) => {
                        for (i, r) in results.into_iter().enumerate() {
                            if let Some(g) = r {
                                faces[i].gaze = Some(g);
                            }
                        }
                    }
                    Err(e) => {
                        warn!(
                            "FaceAnalyzer: batch gaze failed, falling back to per-face: {}",
                            e
                        );
                        for (idx, face) in faces.iter_mut().enumerate() {
                            match gz.predict(image, face) {
                                Ok(g) => face.gaze = Some(g),
                                Err(e2) => warn!(
                                    "FaceAnalyzer: face {} gaze prediction failed: {}",
                                    idx + 1,
                                    e2
                                ),
                            }
                        }
                    }
                }
            }
        }

        // 3g'. 头部姿态批量 (多脸): 优先模型, 其次 FaceMesh 几何回退
        if !single_face {
            if self.config.enable_headpose {
                if let Some(hp) = &mut self.headpose_net {
                    let refs: Vec<&DetectedFace> = faces.iter().collect();
                    match hp.predict_batch(image, &refs) {
                        Ok(results) => {
                            for (i, r) in results.into_iter().enumerate() {
                                if let Some(pose) = r {
                                    faces[i].head_pose = Some(pose);
                                }
                            }
                        }
                        Err(e) => {
                            warn!(
                                "FaceAnalyzer: batch head pose failed, falling back to per-face: {}",
                                e
                            );
                            for (idx, face) in faces.iter_mut().enumerate() {
                                match hp.predict(image, face) {
                                    Ok(r) => face.head_pose = Some(r),
                                    Err(e2) => warn!(
                                        "FaceAnalyzer: face {} head pose model failed: {}",
                                        idx + 1,
                                        e2
                                    ),
                                }
                            }
                        }
                    }
                }
                for (i, face) in faces.iter_mut().enumerate() {
                    if face.head_pose.is_none() {
                        if let Some(mr) = &mesh_results[i] {
                            let pose = pipeline::headpose::estimate_from_mesh(
                                &mr.landmarks_3d,
                                img_w,
                                img_h,
                            );
                            info!(
                                "FaceAnalyzer: face {} head pose from mesh (fallback)",
                                i + 1
                            );
                            face.head_pose = Some(pose);
                        } else if self.headpose_net.is_none() {
                            warn!(
                                "FaceAnalyzer: face {} head pose skipped (no model, no mesh)",
                                i + 1
                            );
                        }
                    }
                }
            }
        } else {
            // 单脸: 模型逐脸 + FaceMesh 几何回退
            if self.config.enable_headpose {
                for (idx, face) in faces.iter_mut().enumerate() {
                    if let Some(hp) = &mut self.headpose_net {
                        match hp.predict(image, face) {
                            Ok(r) => face.head_pose = Some(r),
                            Err(e) => warn!(
                                "FaceAnalyzer: face {} head pose model failed: {}",
                                idx + 1,
                                e
                            ),
                        }
                    }
                    if face.head_pose.is_none() {
                        if let Some(mr) = &mesh_results[idx] {
                            let pose = pipeline::headpose::estimate_from_mesh(
                                &mr.landmarks_3d,
                                img_w,
                                img_h,
                            );
                            info!(
                                "FaceAnalyzer: face {} head pose from mesh (fallback)",
                                idx + 1
                            );
                            face.head_pose = Some(pose);
                        } else if self.headpose_net.is_none() {
                            warn!(
                                "FaceAnalyzer: face {} head pose skipped (no model, no mesh)",
                                idx + 1
                            );
                        }
                    }
                }
            }
        }

        // 3c'. 年龄/性别批量 (多脸)
        if !single_face {
            if let Some(ag) = &mut self.age_gender {
                let refs: Vec<&DetectedFace> = faces.iter().collect();
                match ag.predict_batch(image, &refs) {
                    Ok(results) => {
                        for (i, r) in results.into_iter().enumerate() {
                            if let Some(demo) = r {
                                faces[i].gender = Some(demo.gender);
                                faces[i].age = demo.age;
                            }
                        }
                    }
                    Err(e) => {
                        warn!(
                            "FaceAnalyzer: batch age/gender failed, falling back to per-face: {}",
                            e
                        );
                        for (idx, face) in faces.iter_mut().enumerate() {
                            match ag.predict(image, face) {
                                Ok(demo) => {
                                    face.gender = Some(demo.gender);
                                    face.age = demo.age;
                                }
                                Err(e2) => warn!(
                                    "FaceAnalyzer: face {} AgeGender prediction failed: {}",
                                    idx + 1,
                                    e2
                                ),
                            }
                        }
                    }
                }
            }
        }

        // 3e'. 人脸状态批量
        if let Some(fa) = &mut self.face_attrib {
            if !single_face {
                let refs: Vec<&DetectedFace> = faces.iter().collect();
                match fa.predict_batch(image, &refs) {
                    Ok(results) => {
                        for (i, r) in results.into_iter().enumerate() {
                            if let Some(fs) = r {
                                faces[i].face_state = Some(fs);
                            }
                        }
                    }
                    Err(e) => {
                        warn!(
                            "FaceAnalyzer: batch face attrib failed, falling back to per-face: {}",
                            e
                        );
                        for (idx, face) in faces.iter_mut().enumerate() {
                            match fa.predict(image, face) {
                                Ok(fs) => face.face_state = Some(fs),
                                Err(e2) => warn!(
                                    "FaceAnalyzer: face {} FaceAttribNet prediction failed: {}",
                                    idx + 1,
                                    e2
                                ),
                            }
                        }
                    }
                }
            }
        }

        // 3h'. 质量评估批量 (仅 5 点对齐)
        if let Some(q) = &mut self.quality {
            let indices: Vec<usize> = faces
                .iter()
                .enumerate()
                .filter(|(_, f)| f.landmarks.is_some())
                .map(|(i, _)| i)
                .collect();
            if !indices.is_empty() {
                let batch_out = {
                    let refs: Vec<&DetectedFace> = indices.iter().map(|&i| &faces[i]).collect();
                    match q.predict_batch(image, &refs) {
                        Ok(results) => Some(results),
                        Err(e) => {
                            warn!(
                                "FaceAnalyzer: batch quality failed, falling back to per-face: {}",
                                e
                            );
                            None
                        }
                    }
                };
                match batch_out {
                    Some(results) => {
                        for (k, r) in results.into_iter().enumerate() {
                            if let Some(qr) = r {
                                let idx = indices[k];
                                faces[idx].quality = Some(qr.score);
                            }
                        }
                    }
                    None => {
                        for (idx, face) in faces.iter_mut().enumerate() {
                            if let Some(landmarks) = &face.landmarks {
                                match q.predict(image, landmarks) {
                                    Ok(qr) => face.quality = Some(qr.score),
                                    Err(e2) => warn!(
                                        "FaceAnalyzer: face {} quality prediction failed: {}",
                                        idx + 1,
                                        e2
                                    ),
                                }
                            }
                        }
                    }
                }
            }
        } else if self.config.enable_quality && !supports_alignment {
            // 对齐 supports_alignment: BlazeFace 下无法运行质量评估
            for idx in 0..faces.len() {
                warn!(
                    "FaceAnalyzer: face {} quality skipped (detector lacks 5-pt alignment landmarks)",
                    idx + 1
                );
            }
        }

        info!("FaceAnalyzer: analysis complete — {} face(s)", faces.len());
        Ok(faces)
    }

    /// 检测 + BYTETracker 追踪
    ///
    /// 返回当前存活轨迹 (track_id + bbox + 置信度)。
    pub fn track(&mut self, image: &DynamicImage) -> Result<Vec<pipeline::tracking::TrackResult>> {
        // 检测失败仅 log warn, 返回当前存活轨迹 (对齐 analyze 的 warn 语义)
        let detections = self.detector.detect(image).unwrap_or_else(|e| {
            warn!(
                "FaceAnalyzer: face detection failed, tracking skipped: {}",
                e
            );
            Vec::new()
        });
        match &mut self.tracker {
            Some(tracker) => Ok(tracker.update(&detections)),
            None => Ok(Vec::new()),
        }
    }

    /// 人像抠图 (整图独立运行, 不依赖检测)
    pub fn matting(&mut self, image: &DynamicImage) -> Result<crate::types::MatteResult> {
        match &mut self.modnet {
            Some(mn) => Ok(mn.predict(image)?),
            None => {
                return Err(FaceError::NotEnabled(
                    "Matting not enabled (EngineConfig.enable_matting)".into(),
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 构造一个合成测试图像 (纯色渐变, 不含真实人脸)
    fn synthetic_image(w: u32, h: u32) -> DynamicImage {
        image::RgbImage::from_fn(w, h, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, 128])
        })
        .into()
    }

    /// ORT 会话可用性探测 — 环境内的 ONNX Runtime 可能过旧 (如 Windows 自带 1.17,
    /// ort 2.0-rc.15 需要 >= 1.22) 或未安装; 不可用时跳过依赖真实会话的测试。
    fn ort_available() -> bool {
        crate::models::get_model_bytes("blazeface_128")
            .and_then(|bytes| InferenceSession::from_bytes("blazeface_128", bytes).ok())
            .is_some()
    }

    /// BlazeFace 下 recognition 被 warn-skip: 构造成功 (识别器为 None),
    /// analyze 正常返回且不 panic (占位模型推理失败被 catch 住)
    #[test]
    fn test_blazeface_recognition_warn_skip() {
        if !ort_available() {
            eprintln!("SKIP: no usable ONNX Runtime (set ORT_LIB_PATH to a >=1.22 onnxruntime)");
            return;
        }
        // enable_recognition=true 但 BlazeFace 无 5 点对齐 → 识别器被禁用
        let config = EngineConfig {
            detector_kind: DetectorKind::BlazeFace,
            enable_facemesh: true,
            enable_recognition: true,
            enable_headpose: true,
            enable_quality: true, // 同样要求对齐, 应被 warn-skip
            ..Default::default()
        };

        let mut analyzer =
            FaceAnalyzer::new(config).expect("FaceAnalyzer::new should succeed under BlazeFace");

        let img = synthetic_image(64, 64);
        let faces = analyzer
            .analyze(&img)
            .expect("analyze should not panic under BlazeFace (predictor errors are warnings)");
        // 无真实人脸 + 占位模型 → 无人脸或空 landmarks, 均为合法结果
        assert!(
            faces.iter().all(|f| f.embedding.is_none()),
            "BlazeFace must never produce embeddings (warn-skip semantics)"
        );
    }

    /// SCRFD 配置组合: 全部预测器启用, analyze / track / matting 不 panic
    #[test]
    fn test_scrfd_full_pipeline_no_panic() {
        if !ort_available() {
            eprintln!("SKIP: no usable ONNX Runtime (set ORT_LIB_PATH to a >=1.22 onnxruntime)");
            return;
        }
        let config = EngineConfig {
            detector_kind: DetectorKind::Scrfd,
            enable_facemesh: true,
            enable_recognition: true,
            enable_attributes: true,
            enable_gaze: true,
            enable_headpose: true,
            enable_quality: true,
            enable_spoofing: true,
            enable_matting: true,
            enable_tracking: true,
            max_faces: 4,
            ..Default::default()
        };

        let mut analyzer = FaceAnalyzer::new(config)
            .expect("FaceAnalyzer::new should succeed with SCRFD + all predictors");

        let img = synthetic_image(96, 96);

        // analyze: 逐项 enrich 失败仅 log warn, 整体不 panic
        let faces = analyzer
            .analyze(&img)
            .expect("analyze should return Ok even when placeholder models fail");
        assert!(faces.len() <= 4, "max_faces=4 must truncate the result");

        // track: 检测 + BYTETracker.update
        let _tracks = analyzer.track(&img).expect("track should not panic");

        // matting: 整图独立运行 (占位模型失败时返回 Err 而非 panic)
        let _ = analyzer.matting(&img);
    }

    /// max_faces 截断逻辑: 选项覆盖与配置默认值
    #[test]
    fn test_analyze_max_faces_option() {
        if !ort_available() {
            eprintln!("SKIP: no usable ONNX Runtime (set ORT_LIB_PATH to a >=1.22 onnxruntime)");
            return;
        }
        let config = EngineConfig {
            detector_kind: DetectorKind::Scrfd,
            enable_facemesh: false,
            enable_recognition: false,
            max_faces: 1,
            ..Default::default()
        };
        let mut analyzer = FaceAnalyzer::new(config).expect("construction ok");

        let img = synthetic_image(64, 64);
        let opts = AnalysisOptions { max_faces: 0 }; // 0 → 回退到 config.max_faces=1
        let faces = analyzer
            .analyze_with_options(&img, &opts)
            .expect("analyze_with_options ok");
        assert!(faces.len() <= 1, "config max_faces=1 must be honored");
    }
}
