// ====================================================================
// ONNX 推理引擎 — 封装 ort 2.0.0-rc.15 (load-dynamic 模式)
// ====================================================================
//
// 使用 load-dynamic 特性, 在运行时动态加载 ONNX Runtime 库:
//   - Android: libonnxruntime.so (放在 jniLibs 目录)
//   - Windows: onnxruntime.dll (放在 exe 同目录)
//   - Linux:   libonnxruntime.so (放在 LD_LIBRARY_PATH)
//   - macOS:   libonnxruntime.dylib (放在 @rpath)
//
// 库路径查找顺序:
//   1. 环境变量 ORT_LIB_PATH
//   2. 可执行文件同目录
//   3. 系统默认路径
//
use anyhow::Result;
use log::info;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

/// 全局 ONNX Runtime 环境 (进程内单例)
static ENV: OnceLock<ort::environment::Environment> = OnceLock::new();
static ENV_INIT_ERROR: Mutex<Option<String>> = Mutex::new(None);

/// 获取全局 ONNX Runtime 环境 (首次调用时初始化)
fn get_env() -> Result<&'static ort::environment::Environment> {
    if let Some(env) = ENV.get() {
        return Ok(env);
    }

    // 检查是否之前初始化失败
    {
        let err = ENV_INIT_ERROR.lock().unwrap();
        if let Some(ref e) = *err {
            anyhow::bail!("ONNX Runtime initialization failed previously: {}", e);
        }
    }

    // Android: 静态链接 ORT, 无需动态加载
    #[cfg(target_os = "android")]
    let result = {
        ort::environment::init()
            .with_name("face-core")
            .build()
            .map_err(|e| {
                let msg = format!("Failed to build ONNX Runtime environment: {}", e);
                *ENV_INIT_ERROR.lock().unwrap() = Some(msg.clone());
                anyhow::anyhow!("{}", msg)
            })?
    };

    // 桌面平台: 动态加载 ORT 库
    #[cfg(not(target_os = "android"))]
    let result = {
        let lib_path = find_ort_library();
        info!("Loading ONNX Runtime from: {}", lib_path.display());
        ort::init_from(&lib_path)
            .map_err(|e| {
                let msg = format!("Failed to load ONNX Runtime library: {}", e);
                *ENV_INIT_ERROR.lock().unwrap() = Some(msg.clone());
                anyhow::anyhow!("{}", msg)
            })?
            .with_name("face-core")
            .build()
            .map_err(|e| {
                let msg = format!("Failed to build ONNX Runtime environment: {}", e);
                *ENV_INIT_ERROR.lock().unwrap() = Some(msg.clone());
                anyhow::anyhow!("{}", msg)
            })?
    };

    match ENV.set(result) {
        Ok(()) => {
            info!("ONNX Runtime environment initialized successfully");
            Ok(ENV.get().unwrap())
        }
        Err(_) => {
            // 另一个线程已经初始化了
            Ok(ENV.get().unwrap())
        }
    }
}

/// 查找 ONNX Runtime 库文件路径 (仅动态加载模式使用)
#[cfg(not(target_os = "android"))]
fn find_ort_library() -> PathBuf {
    // 1. 环境变量
    if let Ok(path) = std::env::var("ORT_LIB_PATH") {
        let p = PathBuf::from(&path);
        if p.exists() {
            return p;
        }
    }

    // 2. 可执行文件同目录
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            let candidates = [
                "onnxruntime.dll",
                "libonnxruntime.so",
                "libonnxruntime.dylib",
            ];
            for name in &candidates {
                let p = exe_dir.join(name);
                if p.exists() {
                    return p;
                }
            }
        }
    }

    // 3. 当前工作目录
    let cwd_candidates = [
        "onnxruntime.dll",
        "libonnxruntime.so",
        "libonnxruntime.dylib",
    ];
    for name in &cwd_candidates {
        let p = PathBuf::from(name);
        if p.exists() {
            return p;
        }
    }

    // 4. 默认: 返回库名 (让系统动态加载器查找)
    #[cfg(target_os = "windows")]
    {
        PathBuf::from("onnxruntime.dll")
    }
    #[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
    {
        PathBuf::from("libonnxruntime.so")
    }
    #[cfg(target_os = "macos")]
    {
        PathBuf::from("libonnxruntime.dylib")
    }
}

/// ONNX 推理会话
pub struct InferenceSession {
    name: String,
    session: Session,
}

impl InferenceSession {
    /// 从嵌入式字节创建推理会话
    ///
    /// 大模型用全部核心; 小输入模型 (<128x128) 多线程同步开销超过收益, 用少量线程
    pub fn from_bytes(name: &str, model_bytes: &[u8]) -> Result<Self> {
        let threads = match name {
            "emotion_ferplus" => 2,
            _ => num_cpus(),
        };
        Self::from_bytes_with_threads(name, model_bytes, threads)
    }

    /// 从嵌入式字节创建推理会话, 指定 intra-op 线程数
    pub fn from_bytes_with_threads(name: &str, model_bytes: &[u8], threads: usize) -> Result<Self> {
        info!(
            "Loading model '{}' from embedded bytes ({:.1} KB, intra_threads={})...",
            name,
            model_bytes.len() as f64 / 1024.0,
            threads
        );

        let env = get_env()?;

        let session = Session::builder(env)
            .map_err(|e| anyhow::anyhow!("Failed to create SessionBuilder: {}", e))?
            // ORT >= 1.20 移除了 ORT_ENABLE_LAYOUT (旧 Level3); All 即原全量优化
            .with_optimization_level(GraphOptimizationLevel::All)
            .map_err(|e| anyhow::anyhow!("Failed to set optimization level: {}", e))?
            .with_intra_threads(threads)
            .map_err(|e| anyhow::anyhow!("Failed to set thread count: {}", e))?
            .commit_from_memory(model_bytes)
            .map_err(|e| anyhow::anyhow!("Failed to load model from embedded bytes: {}", e))?;

        info!("Model '{}' loaded successfully", name);

        Ok(Self {
            name: name.to_string(),
            session,
        })
    }

    /// 获取 ort 会话引用
    pub fn session(&self) -> &Session {
        &self.session
    }

    /// 获取 ort 会话可变引用
    pub fn session_mut(&mut self) -> &mut Session {
        &mut self.session
    }

    /// 获取模型名称
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// 获取合理的线程数
fn num_cpus() -> usize {
    #[cfg(target_os = "android")]
    {
        4
    }
    #[cfg(not(target_os = "android"))]
    {
        num_cpus::get().max(1)
    }
}

/// 人脸检测器类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DetectorKind {
    /// BlazeFace 短距离检测器 (128x128, 6 关键点, 不支持 5 点对齐)
    #[default]
    BlazeFace,
    /// SCRFD 500M 检测器 (640x640, 5 点对齐关键点, supports_alignment)
    Scrfd,
}

/// 美颜引擎配置
///
/// 新增可选功能开关默认 false 以控制内嵌模型的二进制体积;
/// 识别 (enable_recognition) 默认 true, 对齐统一分析器的默认行为
/// (BlazeFace 检测器下会因缺少 5 点对齐关键点而 warn-skip)。
#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub enable_detection: bool,
    pub enable_facemesh: bool,
    pub enable_parsing: bool,
    pub detection_threshold: f32,
    pub use_gpu: bool,
    /// 检测器类型 (BlazeFace 或 SCRFD)
    pub detector_kind: DetectorKind,
    /// 人脸识别嵌入提取 (需 5 点对齐 landmarks, 即 SCRFD)
    pub enable_recognition: bool,
    /// 属性四合一: 年龄 / 性别 / 情绪 / 人脸状态
    pub enable_attributes: bool,
    /// 视线估计 (MobileNetV2 Gaze)
    pub enable_gaze: bool,
    /// 头部姿态 (ResNet18 模型 + FaceMesh 几何回退)
    pub enable_headpose: bool,
    /// 人脸图像质量评估 (eDifFIQA, 需 5 点对齐 landmarks)
    pub enable_quality: bool,
    /// 活体检测 (MiniFASNet)
    pub enable_spoofing: bool,
    /// 人像抠图 (MODNet, 整图独立运行)
    pub enable_matting: bool,
    /// 多目标追踪 (BYTETracker, track() 时使用)
    pub enable_tracking: bool,
    /// 最大人脸数, 0 = 不限制 (超出按置信度降序截断)
    pub max_faces: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            enable_detection: true,
            enable_facemesh: true,
            enable_parsing: true,
            detection_threshold: 0.5,
            use_gpu: false,
            detector_kind: DetectorKind::BlazeFace,
            enable_recognition: true,
            enable_attributes: false,
            enable_gaze: false,
            enable_headpose: false,
            enable_quality: false,
            enable_spoofing: false,
            enable_matting: false,
            enable_tracking: false,
            max_faces: 0,
        }
    }
}
