// ====================================================================
// face-cli — 人脸分析命令行工具
//
// 三种运行模式:
//   1. 默认模式: 完整管线演示 (检测/网格/分割/识别/属性/视线/
//      姿态/质量/活体/追踪/抠图)
//   2. --bench: 输出机器可读 JSON (供基准对比脚本消费)
//   3. --render <目录>: 可视化结果输出为图片
// ====================================================================

use anyhow::Result;
use face_core::inference::{DetectorKind, EngineConfig};
use face_core::pipeline::tracking::TrackResult;
use face_core::{FaceAnalyzer, FaceEngine};

fn print_usage() {
    // 使用说明
    println!("Usage: face <image_path>");
    println!("Example: face photo.jpg");
    println!();
    println!("Demo runs the full FaceAnalyzer pipeline (SCRFD + all predictors).");
    println!("Without arguments, only this usage message is printed.");
}

/// 打印单张人脸的全部分析结果 (仅打印存在的字段)
fn print_face(idx: usize, face: &face_core::types::DetectedFace) {
    println!(
        "\n  Face #{}: bbox=({:.0}, {:.0}, {:.0}, {:.0}), confidence={:.4}",
        idx + 1,
        face.bbox.x1,
        face.bbox.y1,
        face.bbox.x2,
        face.bbox.y2,
        face.confidence
    );

    // 追踪 ID
    if let Some(track_id) = face.track_id {
        println!("    Track ID: {}", track_id);
    }

    // 5 点关键点
    if let Some(ref landmarks) = face.landmarks {
        println!("    Landmarks: {} points", landmarks.nrows());
    }

    // 三维网格
    if let Some(score) = face.mesh_score {
        println!("    FaceMesh: 3D mesh, score={:.4}", score);
        if let Some(ref mesh) = face.mesh_3d {
            let pts: Vec<String> = mesh
                .axis_iter(ndarray::Axis(0))
                .take(3)
                .map(|r| format!("({:.3},{:.3})", r[[0]], r[[1]]))
                .collect();
            println!("    Mesh pts[0..3]: {}", pts.join(" "));
        }
    }

    // 识别特征向量
    if let Some(ref embedding) = face.embedding {
        let first5: Vec<String> = embedding
            .iter()
            .take(5)
            .map(|v| format!("{:.4}", v))
            .collect();
        println!(
            "    Embedding: {}-D vector, first5=[{}]",
            embedding.len(),
            first5.join(", ")
        );
    }

    // 年龄与性别
    if let Some(gender) = face.gender {
        let label = match gender {
            face_core::types::Gender::Female => "Female",
            face_core::types::Gender::Male => "Male",
        };
        let age_part = face.age.map(|a| format!(", age={}", a)).unwrap_or_default();
        println!("    Gender: {}{}", label, age_part);
    } else if let Some(age) = face.age {
        println!("    Age: {}", age);
    }

    // 情绪
    if let Some(emotion) = face.emotion {
        let conf = face
            .emotion_confidence
            .map(|c| format!(" (confidence={:.4})", c))
            .unwrap_or_default();
        println!("    Emotion: {}{}", emotion.label(), conf);
    }

    // 人脸状态五项概率 (睁眼/眼镜/口罩/墨镜)
    if let Some(ref fs) = face.face_state {
        println!(
            "    Face state: left_eye_open={:.3}, right_eye_open={:.3}, eyeglasses={:.3}, mask={:.3}, sunglasses={:.3}",
            fs.left_eye_open, fs.right_eye_open, fs.eyeglasses, fs.mask, fs.sunglasses
        );
    }

    // 视线方向
    if let Some(ref gaze) = face.gaze {
        println!(
            "    Gaze: pitch={:.3} rad, yaw={:.3} rad",
            gaze.pitch, gaze.yaw
        );
    }

    // 头部姿态
    if let Some(ref pose) = face.head_pose {
        println!(
            "    Head pose: pitch={:.3}, yaw={:.3}, roll={:.3} rad",
            pose.pitch, pose.yaw, pose.roll
        );
    }

    // 图像质量分
    if let Some(quality) = face.quality {
        println!("    Quality: {:.4}", quality);
    }

    // 活体检测
    if let Some(ref spoofing) = face.spoofing {
        println!(
            "    Spoofing: {} (confidence={:.4})",
            if spoofing.is_real { "real" } else { "spoof" },
            spoofing.confidence
        );
    }
}

/// 打印当前存活的追踪轨迹
fn print_tracks(tracks: &[TrackResult]) {
    println!("\n--- Tracking Results ---");
    if tracks.is_empty() {
        println!("No active tracks");
        return;
    }
    for t in tracks {
        println!(
            "  Track #{}: bbox=({:.0}, {:.0}, {:.0}, {:.0}), confidence={:.4}",
            t.track_id, t.bbox.x1, t.bbox.y1, t.bbox.x2, t.bbox.y2, t.confidence
        );
    }
}

fn main() -> Result<()> {
    face_core::utils::common::init_logging();
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        print_usage();
        return Ok(());
    }
    let image_path = &args[1];

    // bench 模式: 对单张图输出机器可读 JSON
    if args.get(2).map(|s| s.as_str()) == Some("--bench") {
        return run_bench(image_path);
    }

    // render 模式: 可视化结果输出图片 (边界框/关键点/网格/解析叠加)
    if let Some(flag) = args.get(2) {
        if flag == "--render" {
            let out_dir = args.get(3).cloned().unwrap_or_else(|| "output".to_string());
            return run_render(image_path, &out_dir);
        }
    }

    println!("=== face-core: Cross-Platform Face Analysis Engine ===");

    // 初始化引擎一: 检测 + 三维网格 + 语义分割
    let engine_config = EngineConfig::default();
    let mut engine = FaceEngine::new(engine_config)?;
    println!("FaceEngine initialized (detector + facemesh + parsing)");

    // 初始化引擎二: 检测 + 全部预测器 (识别/属性/视线/姿态/质量/活体/追踪/抠图)
    let analyzer_config = EngineConfig {
        detector_kind: DetectorKind::Scrfd,
        enable_facemesh: true,
        enable_parsing: false, // 分割由引擎一负责, 避免重复计算
        enable_recognition: true,
        enable_attributes: true,
        enable_gaze: true,
        enable_headpose: true,
        enable_quality: true,
        enable_spoofing: true,
        enable_matting: true,
        enable_tracking: true,
        ..Default::default()
    };
    let mut analyzer = FaceAnalyzer::new(analyzer_config)?;
    println!("FaceAnalyzer initialized (SCRFD + recognition + predictors + tracking + matting)");

    println!("Loading image: {}", image_path);
    let image = engine.load_image(image_path)?;
    println!("Image dimensions: {}x{}", image.width(), image.height());

    // 统一分析: 一次调用产出全部人脸属性
    println!("\n--- FaceAnalyzer Results ---");
    let faces = analyzer.analyze(&image)?;
    println!("Detected {} face(s)", faces.len());
    for (i, face) in faces.iter().enumerate() {
        print_face(i, face);
    }

    // 调试: 落盘对齐后的 112x112 人脸, 用于像素级核对
    if let Some(face) = faces.first() {
        if let Some(ref landmarks) = face.landmarks {
            let pts: Vec<(f32, f32)> = landmarks
                .axis_iter(ndarray::Axis(0))
                .map(|r| (r[[0]], r[[1]]))
                .collect();
            println!("RUST landmarks = {:?}", pts);
            if let Some(pts5) = <[(f32, f32); 5]>::try_from(pts).ok() {
                let (fwd, _inv) = face_core::utils::alignment::estimate_norm_112(&pts5);
                println!("RUST fwd = {:?}", fwd);
                let aligned = face_core::pipeline::recognition::align_face(&image, landmarks, 112)?;
                aligned.save("target/aligned_debug.png").ok();
            }
            // 落盘 letterbox 画布, 用于核对缩放填充参数
            let (canvas, scale, pad_x, pad_y) =
                face_core::utils::image_utils::letterbox(&image, 640, 640);
            canvas.save("target/letterbox_debug.png").ok();
            println!("RUST letterbox: scale={} pad=({},{})", scale, pad_x, pad_y);
            if let Some(ref emb) = face.embedding {
                std::fs::write(
                    "target/embedding_debug.bin",
                    emb.iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<u8>>(),
                )
                .ok();
            }
        }
    }

    // 引擎一分析: 输出网格与解析遮罩统计
    println!("\n--- FaceEngine Results ---");
    let beauty = engine.analyze(&image)?;
    println!("Detected {} face(s)", result.faces.len());
    for (i, face) in result.faces.iter().enumerate() {
        println!(
            "\n  Face #{}: bbox=({:.0}, {:.0}, {:.0}, {:.0}), confidence={:.4}",
            i + 1,
            face.face.bbox.x1,
            face.face.bbox.y1,
            face.face.bbox.x2,
            face.face.bbox.y2,
            face.face.confidence
        );

        if let Some(ref mesh) = face.mesh_result {
            println!("    FaceMesh: 468 3D points, score={:.4}", mesh.score);
        }

        if let Some(ref mask) = face.parsing_mask {
            println!("    Parsing: {}x{} 19-class mask", mask.width, mask.height);
            let skin_count = mask.get_skin_mask().iter().filter(|&&p| p == 1).count();
            let lips_count = mask.get_lips_mask().iter().filter(|&&p| p == 1).count();
            let eyes_count = mask.get_eyes_mask().iter().filter(|&&p| p == 1).count();
            println!(
                "      skin={}, lips={}, eyes={} pixels",
                skin_count, lips_count, eyes_count
            );
        }
    }

    // 多目标追踪
    let tracks = analyzer.track(&image)?;
    print_tracks(&tracks);

    // 人像抠图
    println!("\n--- Matting Results ---");
    match analyzer.matting(&image) {
        Ok(matte) => {
            println!(
                "Matte: {}x{} alpha mask (mean={:.4})",
                matte.width,
                matte.height,
                matte.alpha.iter().sum::<f32>() / matte.alpha.len().max(1) as f32
            );
        }
        Err(e) => println!("Matting skipped: {}", e),
    }

    Ok(())
}

/// bench 模式: 对单张图输出机器可读 JSON
fn run_bench(image_path: &str) -> Result<()> {
    let config = EngineConfig {
        detector_kind: DetectorKind::Scrfd,
        enable_facemesh: true,
        enable_parsing: true,
        enable_recognition: true,
        enable_attributes: true,
        enable_gaze: true,
        enable_headpose: true,
        enable_quality: true,
        enable_spoofing: true,
        enable_matting: true,
        enable_tracking: true,
        ..Default::default()
    };
    let mut analyzer = FaceAnalyzer::new(config)?;
    let image = face_core::utils::image_utils::load_image_with_exif(image_path)?;
    let faces = analyzer.analyze(&image)?;

    let face_json: Vec<serde_json::Value> = faces
        .iter()
        .map(|f| {
            let mut o = serde_json::json!({
                "bbox": [f.bbox.x1, f.bbox.y1, f.bbox.x2, f.bbox.y2],
                "confidence": f.confidence,
            });
            if let Some(ref lm) = f.landmarks {
                let pts: Vec<[f32; 2]> = lm
                    .axis_iter(ndarray::Axis(0))
                    .map(|r| [r[[0]], r[[1]]])
                    .collect();
                o["landmarks"] = serde_json::json!(pts);
            }
            if let Some(g) = f.gender {
                o["gender"] = serde_json::json!(format!("{:?}", g).to_lowercase());
            }
            if let Some(a) = f.age {
                o["age"] = serde_json::json!(a);
            }
            if let Some(e) = f.emotion {
                o["emotion"] = serde_json::json!(format!("{:?}", e).to_lowercase());
            }
            if let Some(c) = f.emotion_confidence {
                o["emotion_confidence"] = serde_json::json!(c);
            }
            if let Some(ref s) = f.face_state {
                o["face_state"] = serde_json::json!({
                    "left_eye_open": s.left_eye_open,
                    "right_eye_open": s.right_eye_open,
                    "eyeglasses": s.eyeglasses,
                    "mask": s.mask,
                    "sunglasses": s.sunglasses,
                });
            }
            if let Some(q) = f.quality {
                o["quality"] = serde_json::json!(q);
            }
            if let Some(ref g) = f.gaze {
                o["gaze"] = serde_json::json!({"pitch": g.pitch, "yaw": g.yaw});
            }
            if let Some(ref h) = f.head_pose {
                o["head_pose"] =
                    serde_json::json!({"pitch": h.pitch, "yaw": h.yaw, "roll": h.roll});
            }
            if let Some(ref s) = f.spoofing {
                o["spoofing"] =
                    serde_json::json!({"is_real": s.is_real, "confidence": s.confidence});
            }
            if let Some(ref e) = f.embedding {
                let cos = 0.0f32;
                o["embedding"] =
                    serde_json::json!(e.iter().map(|v| (v + cos) as f32).collect::<Vec<f32>>());
            }
            if let Some(ref m) = f.mesh_3d {
                o["mesh_points"] = serde_json::json!(m.nrows());
                o["mesh_score"] = serde_json::json!(f.mesh_score.unwrap_or(0.0));
            }
            o
        })
        .collect();

    let out = serde_json::json!({
        "image": image_path,
        "faces": face_json,
    });
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}

/// render 模式: 检测可视化 + 网格 + 解析叠加, 输出图片文件
fn run_render(image_path: &str, out_dir: &str) -> Result<()> {
    use face_core::pipeline::render as render_mod;

    std::fs::create_dir_all(out_dir)?;

    // 两个引擎分工: 前者负责检测/网格/抠图, 后者负责语义分割
    let mut analyzer = FaceAnalyzer::new(EngineConfig {
        detector_kind: DetectorKind::Scrfd,
        enable_facemesh: true,
        enable_matting: true,
        ..Default::default()
    })?;
    let mut engine = FaceEngine::new(EngineConfig {
        detector_kind: DetectorKind::Scrfd,
        enable_facemesh: true,
        enable_parsing: true,
        ..Default::default()
    })?;
    let image = face_core::utils::image_utils::load_image_with_exif(image_path)?;
    let faces = analyzer.analyze(&image)?;
    let result = engine.analyze(&image)?;
    let stem = std::path::Path::new(image_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("image")
        .to_string();
    let rgb = image.to_rgb8();

    // 1) 绘制检测框 + 关键点 + 网格点
    let overlay = render_mod::draw_faces(&rgb, &faces);
    let out1 = format!("{}/{}_det.jpg", out_dir, stem);
    overlay.save(&out1)?;
    println!("saved {}", out1);

    // 2) 解析遮罩叠加 (取第一张有遮罩的人脸)
    if let Some(pm) = result.faces.iter().find_map(|f| f.parsing_mask.as_ref()) {
        let out2 = format!("{}/{}_parsing.jpg", out_dir, stem);
        render_mod::draw_parsing(&rgb, pm).save(&out2)?;
        println!("saved {}", out2);
        let out3 = format!("{}/{}_parsing_solid.png", out_dir, stem);
        render_mod::draw_parsing_solid(pm).save(&out3)?;
        println!("saved {}", out3);
    }

    // 3) 人像抠图 alpha 遮罩
    if let Ok(matte) = analyzer.matting(&image) {
        let out4 = format!("{}/{}_matte.png", out_dir, stem);
        let mut alpha = image::GrayImage::new(matte.width, matte.height);
        for y in 0..matte.height {
            for x in 0..matte.width {
                let v = (matte.alpha[[y as usize, x as usize]] * 255.0).clamp(0.0, 255.0) as u8;
                alpha.put_pixel(x, y, image::Luma([v]));
            }
        }
        alpha.save(&out4)?;
        println!("saved {}", out4);
    }

    Ok(())
}
