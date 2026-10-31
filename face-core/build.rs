// ====================================================================
// build.rs — 构建脚本
// ====================================================================
// 检查模型文件并在缺失时生成最小的占位文件以确保编译通过。
// Checks model files and generates minimal placeholder files when missing.

use std::fs;
use std::path::Path;

fn main() {
    let required_models = [
        ("blazeface_128.onnx", "BlazeFace face detector"),
        ("facemesh_468.onnx", "FaceMesh 468 3D mesh"),
        ("bisenet_r18.onnx", "BiSeNet face parsing"),
        (
            "scrfd_500m_kps.onnx",
            "SCRFD 500M face detector with 5-pt landmarks",
        ),
        ("arcface_mbf.onnx", "ArcFace MobileNetFace recognition"),
        ("genderage.onnx", "AgeGender age/gender predictor"),
        ("affectnet8.onnx", "AffectNet8 emotion recognizer"),
        ("face_attrib_net.onnx", "FaceAttribNet face-state predictor"),
        ("mobilenetv2_gaze.onnx", "MobileNetV2 gaze estimator"),
        ("resnet18_headpose.onnx", "ResNet18 head pose estimator"),
        ("ediffiqa.onnx", "eDifFIQA quality assessor"),
        ("minifasnet.onnx", "MiniFASNet anti-spoofing"),
        ("modnet.onnx", "MODNet portrait matting"),
    ];

    let models_dir = Path::new("models");
    let _ = fs::create_dir_all(models_dir);

    for (filename, desc) in &required_models {
        let path = models_dir.join(filename);
        println!("cargo:rerun-if-changed={}", path.display());

        if path.exists() {
            let size = fs::metadata(&path).unwrap().len();
            if size < 100 {
                println!(
                    "cargo:warning=Model file '{}' appears to be a placeholder ({} bytes)",
                    filename, size
                );
                println!(
                    "cargo:warning=Please download the real model: python scripts/download_models.py"
                );
            }
        } else {
            println!(
                "cargo:warning===== MISSING MODEL: {} ({}) =====",
                filename, desc
            );
            println!("cargo:warning=Generating placeholder file for compilation only.");
            println!(
                "cargo:warning=Please download the real model: python scripts/download_models.py"
            );
            println!("cargo:warning=");

            // 生成最小的占位 ONNX 文件 (仅用于编译通过, 不能运行推理)
            // Generate minimal placeholder ONNX file (for compilation only, won't run inference)
            let placeholder = generate_minimal_onnx(filename);
            fs::write(&path, &placeholder).unwrap();
        }
    }
}

/// 生成最小的合法 ONNX 文件 (单 Add 节点 x+x→y, 能被 ORT 正常加载)
/// Generate a minimal valid ONNX file (single Add node x+x→y, loadable by ORT)
///
/// 与此前"手写魔数"版本不同, 该占位文件是结构合法的 ONNX 模型:
/// 会话可以创建成功, 但不含真实模型的输出节点 — 推理时的输出名不匹配
/// 会作为普通错误被上层 (FaceAnalyzer) catch 住, 仅 log warn 不中断。
/// Unlike the earlier hand-rolled "magic bytes" version, this placeholder is a
/// structurally valid ONNX model: session creation succeeds, but the expected
/// model outputs are absent — output-name mismatches surface as ordinary
/// errors that FaceAnalyzer catches and logs as warnings.
fn generate_minimal_onnx(_name: &str) -> Vec<u8> {
    // 由 onnx==1.23 序列化的等价模型 (onnx.helper.make_model) 固化而来:
    //   ir_version: 9, opset_import: {"" : 17}, producer: "placeholder"
    //   graph: input x:float[1,3,128,128] --Add--> output y:float[1,3,128,128]
    // Fossilized from the equivalent model serialized by onnx==1.23
    // (onnx.helper.make_model), so no protobuf codegen is needed here.
    let proto: &[u8] = &[
        0x08, 0x09, // ir_version = 9
        0x12, 0x0b, // producer_name = "placeholder"
        0x70, 0x6c, 0x61, 0x63, 0x65, 0x68, 0x6f, 0x6c, 0x64, 0x65, 0x72, 0x3a,
        0x51, // graph (field 7, len 0x51)
        0x0a, 0x0e, //   node: name "x" input "x" output "y" op "Add"
        0x0a, 0x01, 0x78, 0x0a, 0x01, 0x78, 0x12, 0x01, 0x79, 0x22, 0x03, 0x41, 0x64, 0x64, 0x12,
        0x01, 0x67, //   graph name "g"
        0x5a, 0x1d, //   input "x": float[1,3,128,128]
        0x0a, 0x01, 0x78, 0x12, 0x18, 0x0a, 0x16, 0x08, 0x01, 0x12, 0x12, 0x0a, 0x02, 0x08, 0x01,
        0x0a, 0x02, 0x08, 0x03, 0x0a, 0x03, 0x08, 0x80, 0x01, 0x0a, 0x03, 0x08, 0x80, 0x01, 0x62,
        0x1d, //   output "y": float[1,3,128,128]
        0x0a, 0x01, 0x79, 0x12, 0x18, 0x0a, 0x16, 0x08, 0x01, 0x12, 0x12, 0x0a, 0x02, 0x08, 0x01,
        0x0a, 0x02, 0x08, 0x03, 0x0a, 0x03, 0x08, 0x80, 0x01, 0x0a, 0x03, 0x08, 0x80, 0x01, 0x42,
        0x04, 0x0a, 0x00, 0x10, 0x11, // opset_import { "": 17 }
    ];

    proto.to_vec()
}
