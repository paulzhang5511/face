// ====================================================================
// 模型嵌入管理 — 将 ONNX 模型文件内嵌到 Rust 二进制中
// ====================================================================
// Android 平台优势: 不需要单独的模型文件下载, 所有模型随 apk 一起发布

use log::info;

// =========================================================================
// 通过 include_bytes! 在编译时将 ONNX 模型文件嵌入到二进制中
// =========================================================================

/// BlazeFace 人脸检测模型 (128x128, ~460KB)
const BLAZEFACE_MODEL: &[u8] = include_bytes!("../../models/blazeface_128.onnx");

/// FaceMesh V1 468点3D密集人脸网格模型 (192x192)
const FACEMESH_468_MODEL: &[u8] = include_bytes!("../../models/facemesh_468.onnx");

/// BiSeNet 人脸语义分割模型 (ResNet18, 19类, 512x512)
const BISENET_MODEL: &[u8] = include_bytes!("../../models/bisenet_r18.onnx");

/// SCRFD 500M 人脸检测模型 (640x640, 5点关键点, supports_alignment)
const SCRFD_MODEL: &[u8] = include_bytes!("../../models/scrfd_500m_kps.onnx");

/// 人脸识别模型 (MobileNetFace, w600k, 112x112, 512维)
const ARCFACE_MBF_MODEL: &[u8] = include_bytes!("../../models/arcface_mbf.onnx");

/// AgeGender 年龄/性别预测模型 (224x224)
const GENDERAGE_MODEL: &[u8] = include_bytes!("../../models/genderage.onnx");

/// FER+ 情绪识别模型 (64x64 灰度, 8类, 动态 batch)
const FERPLUS_MODEL: &[u8] = include_bytes!("../../models/emotion-ferplus-8_batch.onnx");

/// FaceAttribNet 人脸状态模型 (128x128, 5项独立二分类)
const FACE_ATTRIB_NET_MODEL: &[u8] = include_bytes!("../../models/face_attrib_net.onnx");

/// MobileNetV2 视线估计模型 (224x224, 动态 batch)
const MOBILENETV2_GAZE_MODEL: &[u8] = include_bytes!("../../models/mobilenetv2_gaze_batch.onnx");

/// ResNet18 头部姿态估计模型 (224x224, 动态 batch)
const RESNET18_HEADPOSE_MODEL: &[u8] = include_bytes!("../../models/resnet18_headpose_batch.onnx");

/// eDifFIQA 人脸质量评估模型 (112x112, 5点对齐)
const EDIFFIQA_MODEL: &[u8] = include_bytes!("../../models/ediffiqa.onnx");

/// MiniFASNet 活体检测模型 (80x80)
const MINIFASNET_MODEL: &[u8] = include_bytes!("../../models/minifasnet.onnx");

/// MODNet 人像抠图模型 (短边 512)
const MODNET_MODEL: &[u8] = include_bytes!("../../models/modnet.onnx");

/// 模型注册信息
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// 模型名称
    pub name: &'static str,
    /// 模型原始字节
    pub bytes: &'static [u8],
    /// 输入尺寸 (宽, 高)
    pub input_size: (u32, u32),
    /// 输入通道数
    pub input_channels: usize,
    /// 模型描述 (英文)
    pub description: &'static str,
}

/// 所有内嵌模型
pub static EMBEDDED_MODELS: &[ModelInfo] = &[
    ModelInfo {
        name: "blazeface_128",
        bytes: BLAZEFACE_MODEL,
        input_size: (128, 128),
        input_channels: 3,
        description: "BlazeFace short-range face detector, 128x128 RGB input, outputs 6 keypoints",
    },
    ModelInfo {
        name: "facemesh_468",
        bytes: FACEMESH_468_MODEL,
        input_size: (192, 192),
        input_channels: 3,
        description: "MediaPipe FaceMesh V1, 468 3D dense landmarks, 192x192 RGB input",
    },
    ModelInfo {
        name: "bisenet_r18",
        bytes: BISENET_MODEL,
        input_size: (512, 512),
        input_channels: 3,
        description: "BiSeNet ResNet18, 19-class face semantic segmentation, 512x512 RGB input",
    },
    ModelInfo {
        name: "scrfd_500m_kps",
        bytes: SCRFD_MODEL,
        input_size: (640, 640),
        input_channels: 3,
        description: "SCRFD 500M face detector with 5-point alignment landmarks, 640x640 RGB input",
    },
    ModelInfo {
        name: "arcface_mbf",
        bytes: ARCFACE_MBF_MODEL,
        input_size: (112, 112),
        input_channels: 3,
        description: "ArcFace MobileNetFace w600k recognition model, 112x112 RGB input, 512-D embeddings",
    },
    ModelInfo {
        name: "genderage",
        bytes: GENDERAGE_MODEL,
        input_size: (224, 224),
        input_channels: 3,
        description: "AgeGender age/gender prediction model, 224x224 RGB input",
    },
    ModelInfo {
        name: "emotion_ferplus",
        bytes: FERPLUS_MODEL,
        input_size: (64, 64),
        input_channels: 1,
        description: "FER+ emotion recognition model (8 classes), 64x64 grayscale input",
    },
    ModelInfo {
        name: "face_attrib_net",
        bytes: FACE_ATTRIB_NET_MODEL,
        input_size: (128, 128),
        input_channels: 3,
        description: "FaceAttribNet face-state model (5 binary attributes), 128x128 RGB input",
    },
    ModelInfo {
        name: "mobilenetv2_gaze",
        bytes: MOBILENETV2_GAZE_MODEL,
        input_size: (224, 224),
        input_channels: 3,
        description: "MobileNetV2 gaze estimation model, 224x224 RGB input",
    },
    ModelInfo {
        name: "resnet18_headpose",
        bytes: RESNET18_HEADPOSE_MODEL,
        input_size: (224, 224),
        input_channels: 3,
        description: "ResNet18 head pose estimation model, 224x224 RGB input",
    },
    ModelInfo {
        name: "ediffiqa",
        bytes: EDIFFIQA_MODEL,
        input_size: (112, 112),
        input_channels: 3,
        description: "eDifFIQA face image quality assessment, 112x112 RGB input (5-pt aligned)",
    },
    ModelInfo {
        name: "minifasnet",
        bytes: MINIFASNET_MODEL,
        input_size: (80, 80),
        input_channels: 3,
        description: "MiniFASNet anti-spoofing model, 80x80 RGB input",
    },
    ModelInfo {
        name: "modnet",
        bytes: MODNET_MODEL,
        input_size: (512, 512),
        input_channels: 3,
        description: "MODNet portrait matting model, shorter-side 512 RGB input",
    },
];

/// 获取嵌入式模型字节 (按名称)
pub fn get_model_bytes(name: &str) -> Option<&'static [u8]> {
    EMBEDDED_MODELS
        .iter()
        .find(|m| m.name == name)
        .map(|m| m.bytes)
}

/// 获取嵌入式模型信息 (按名称)
pub fn get_model_info(name: &str) -> Option<&'static ModelInfo> {
    EMBEDDED_MODELS.iter().find(|m| m.name == name)
}

/// 打印所有内嵌模型信息 (用于日志)
pub fn log_embedded_models() {
    info!("Embedded models count: {}", EMBEDDED_MODELS.len());
    for model in EMBEDDED_MODELS {
        info!(
            "  [{}] size={:.1}KB, input={}x{}x{}, desc={}",
            model.name,
            model.bytes.len() as f64 / 1024.0,
            model.input_size.0,
            model.input_size.1,
            model.input_channels,
            model.description
        );
    }
}
