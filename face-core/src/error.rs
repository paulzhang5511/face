// ====================================================================
// 统一错误类型 — 库的公共错误接口
// ====================================================================

/// face-core 全部公共 API 的统一错误类型
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum FaceError {
    /// ONNX Runtime 初始化或推理失败
    #[error("ONNX Runtime error: {0}")]
    Onnx(String),

    /// 模型输出缺失或形状不符
    #[error("invalid model output: {0}")]
    InvalidOutput(String),

    /// 输入参数或图像不合法
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// 图像编解码或处理失败
    #[error("image error: {0}")]
    Image(#[from] image::ImageError),

    /// 引擎未启用所需功能 (检查 EngineConfig 的开关)
    #[error("capability not enabled: {0}")]
    NotEnabled(String),

    /// 其他内部错误
    #[error("{0}")]
    Other(String),
}

// 内部使用的 anyhow 错误经此转换进入公共错误类型,
// 保证对外 API 不暴露 anyhow
impl From<anyhow::Error> for FaceError {
    fn from(e: anyhow::Error) -> Self {
        Self::Other(e.to_string())
    }
}

/// 库统一的 Result 别名, 公共 API 一律返回此类型
pub type Result<T> = std::result::Result<T, FaceError>;
