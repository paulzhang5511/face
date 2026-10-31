# face — 跨平台人脸分析引擎

基于 **Rust + ONNX Runtime** 的人脸分析引擎。所有模型在编译期通过 `include_bytes!` 内嵌为二进制，运行时**零外部模型文件依赖**，支持 Windows / Linux / macOS / Android。

提供两套 API：

- **`FaceAnalyzer`** — 统一人脸分析编排器，按 `EngineConfig` 开关逐项加载检测器、识别器与各预测器。
- **`FaceEngine`** — 基础管线（检测 + 3D 网格 + 语义分割），面向快速集成。

---

## 功能一览

| 能力 | 模型 | 说明 |
|---|---|---|
| 人脸检测 | BlazeFace (128×128) | 轻量短距检测，6 关键点，默认检测器 |
| 人脸检测 | SCRFD 500M (640×640) | 5 点对齐关键点，支持识别 / 质量评估 |
| 3D 网格 | FaceMesh (192×192) | 468 点稠密三维网格 |
| 语义分割 | BiSeNet ResNet18 (512×512) | 19 类人脸解析（皮肤 / 头发 / 嘴唇等） |
| 人脸识别 | ArcFace MobileFace (112×112) | 512 维特征向量，1:1 比对 / 1:N 检索 |
| 属性分析 | AgeGender + FER+ + FaceAttribNet | 年龄 / 性别 / 情绪 (8 类) / 人脸状态 (5 属性) |
| 视线估计 | MobileNetV2 (224×224) | 双眼注视点坐标 |
| 头部姿态 | ResNet18 (224×224) | 欧拉角 (pitch/yaw/roll)，FaceMesh 几何回退 |
| 质量评估 | eDifFIQA (112×112) | 人脸图像质量打分（需 5 点对齐） |
| 活体检测 | MiniFASNet (80×80) | 真人脸 / 照片攻击判别 |
| 人像抠图 | MODNet (短边 512) | 高质量 alpha 遮罩，整图独立运行 |
| 多目标追踪 | BYTETracker | Kalman 预测 + IoU/距离级联匹配（视频流） |
| 人脸匿名化 | — | 模糊 / 像素化 / 深黑遮盖 |

---

## 目录结构

```
face/
├── face-core/              # 核心库（crate: face-core）
│   ├── src/
│   │   ├── inference/       # ONNX Runtime 推理封装、EngineConfig、FaceEngine、FaceAnalyzer
│   │   ├── pipeline/        # 各分析模型管线（detector/facemesh/bisenet/recognition/...）
│   │   ├── types.rs         # 统一数据类型（DetectedFace / FaceRegion / 各结果结构）
│   │   └── utils/           # 图像工具 / 人脸对齐 / 公共函数（日志、IoU、NMS）
│   ├── build.rs             # 编译期校验内嵌模型
│   └── models/              # 模型存放目录（DOWNLOAD.md 下载清单，不随仓库分发）
├── face-cli/               # 命令行工具（bin: face）
│   └── src/bin/             # face / face-bench / bench10 / batch-check
├── face-demo/              # 示例集（单二进制 face-demo，含 15 个示例函数）
│   └── assets/             # 随仓库分发的测试图片
├── face-android/           # Android 设备侧基准 / 验证工具
│   └── src/bin/             # face-bench-android / batch-check-android
├── scripts/                # download_models.py / build_android.sh
├── Cargo.toml              # workspace 定义
├── Cargo.lock
├── LICENSE                 # MIT
└── README.md
```

---

## 快速开始

### 1. 下载模型（首次构建必需）

模型不随仓库分发，需先下载到 `face-core/models/`：

```bash
python scripts/download_models.py
```

或参考 [face-core/models/DOWNLOAD.md](face-core/models/DOWNLOAD.md) 手动下载（13 个 ONNX 文件，共约 223 MB，含国内镜像）。

> 注意：`affectnet8.onnx` 在 HF 仓库中名为 `affecnet8.script`（原作者拼写如此），保存时须重命名为 `affectnet8.onnx`。

### 2. 构建

```bash
cargo build --release
```

模型通过 `include_bytes!` 嵌入二进制，构建产物无外部模型文件依赖。运行时通过 `load-dynamic` 加载 ONNX Runtime 动态库，查找顺序：

1. 环境变量 `ORT_LIB_PATH`
2. 可执行文件同目录（`onnxruntime.dll` / `libonnxruntime.so` / `libonnxruntime.dylib`）
3. 当前工作目录
4. 系统动态库搜索路径

### 3. 运行

```bash
# face: 完整管线分析（SCRFD + 全部预测器）
./target/release/face <image_path>

# face: 机器可读 JSON 输出（基准对比用）
./target/release/face <image_path> --bench

# face: 可视化结果输出为图片
./target/release/face <image_path> --render <output_dir>

# face-demo: 运行 15 个示例中的任意一个（测试图已随仓库分发在 face-demo/assets/）
./target/release/face_demo 01_detection
./target/release/face_demo 05_analyzer
```

> Windows 下两个二进制分别为 `face.exe`（CLI）与 `face-demo.exe`（示例集）。

`face-demo` 示例列表（单二进制 `face-demo` 以示例名分发）：

| 示例 | 功能 |
|---|---|
| `01_detection` | 人脸检测 |
| `02_alignment` | 人脸对齐 |
| `03_verification` | 人脸验证（1:1 比对） |
| `04_search` | 人脸搜索（1:N 匹配） |
| `05_analyzer` | 综合人脸分析 |
| `06_parsing` | 人脸解析（19 类分割） |
| `07_anonymization` | 人脸匿名化 |
| `08_gaze` | 视线估计 |
| `09_segmentation` | 人脸分割（抠像换背景） |
| `10_vector_store` | 人脸特征向量库 |
| `11_head_pose` | 头部姿态估计 |
| `12_recognition` | 人脸识别（相似度矩阵） |
| `13_matting` | 人像抠图 |
| `14_attributes` | 人脸属性检测 |
| `15_mesh` | 稠密人脸网格 |

---

## 库使用

### FaceEngine — 基础管线（检测 + 网格 + 分割）

```rust
use face_core::prelude::*;
use face_core::inference::EngineConfig;

let config = EngineConfig::default();
let mut engine = FaceEngine::new(config)?;
let image = engine.load_image("face.jpg")?;
let result = engine.analyze(&image)?;
for face in result.faces {
    println!("bbox={:?}, confidence={:.2}", face.face.bbox, face.face.confidence);
}
```

### FaceAnalyzer — 全功能分析（按需开关）

```rust
use face_core::FaceAnalyzer;
use face_core::inference::{DetectorKind, EngineConfig};

let mut config = EngineConfig::default();
config.detector_kind = DetectorKind::Scrfd;   // 5 点对齐，支持识别 / 质量
config.enable_recognition = true;             // 人脸识别（默认开）
config.enable_attributes = true;              // 年龄 / 性别 / 情绪 / 状态
config.enable_gaze = true;                    // 视线估计
config.enable_headpose = true;                // 头部姿态
config.enable_tracking = true;                // 多目标追踪（track() 时使用）
config.enable_matting = true;                 // 人像抠图（独立运行）

let mut analyzer = FaceAnalyzer::new(config)?;
let faces = analyzer.analyze(&image)?;
for f in &faces {
    if let Some(e) = &f.embedding { /* 512 维特征 */ }
    if let Some(g) = f.gender   { /* 男 / 女 */ }
    if let Some(a) = f.age      { /* 年龄 */ }
}
```

### 视频流追踪

```rust
// 逐帧调用，返回当前存活轨迹（track_id + bbox + 置信度）
let tracks = analyzer.track(&frame)?;
for t in &tracks {
    println!("track_id={:?}, bbox={:?}", t.track_id, t.bbox);
}
```

---

## 配置项

`EngineConfig` 全部字段（见 `face-core/src/inference/mod.rs`）：

| 字段 | 默认值 | 说明 |
|---|---|---|
| `enable_detection` | `true` | 人脸检测 |
| `enable_facemesh` | `true` | 468 点 3D 网格 |
| `enable_parsing` | `true` | 19 类语义分割 |
| `detection_threshold` | `0.5` | 检测置信度阈值 |
| `use_gpu` | `false` | GPU 推理开关 |
| `detector_kind` | `BlazeFace` | 检测器类型（`BlazeFace` / `Scrfd`） |
| `enable_recognition` | `true` | 人脸识别（需 SCRFD 的 5 点对齐） |
| `enable_attributes` | `false` | 年龄 / 性别 / 情绪 / 人脸状态 |
| `enable_gaze` | `false` | 视线估计 |
| `enable_headpose` | `false` | 头部姿态 |
| `enable_quality` | `false` | 质量评估（需 5 点对齐） |
| `enable_spoofing` | `false` | 活体检测 |
| `enable_matting` | `false` | 人像抠图 |
| `enable_tracking` | `false` | BYTETracker 追踪 |
| `max_faces` | `0` | 最大人脸数，0 = 不限制 |

> 可选功能默认关闭以控制内嵌模型的二进制体积；`enable_recognition` 默认开启，但在 BlazeFace 检测器下因缺少 5 点对齐关键点会告警跳过，需切换 `DetectorKind::Scrfd`。

---

## Android 构建

```bash
# 前置：rustup Android targets + cargo-ndk + ANDROID_NDK_HOME
./scripts/build_android.sh release
```

Android 平台日志走 `android_logger`（tag `face-core`），桌面平台走 `env_logger`（环境变量 `RUST_LOG` 控制级别，默认 `info`）。

---

## 测试

```bash
cargo test --workspace
```

---

## 许可证

本项目以 **MIT 许可证** 开源，详见 [LICENSE](LICENSE)。

模型权重来自 Hugging Face 仓库 `yakhyo/uniface-weights`（UniFace 官方权重），其许可以原作者发布为准；本仓库仅提供下载清单，不包含任何模型权重文件。
