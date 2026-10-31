// 验证 gaze/headpose/emotion batch 推理与逐脸推理结果一致 (多脸图)
use face_core::FaceAnalyzer;
use face_core::inference::InferenceSession;
use face_core::pipeline::attribute::{EmotionModel, EmotionNet};
use face_core::pipeline::gaze::GazeNet;
use face_core::pipeline::headpose::HeadPoseNet;
use face_core::prelude::{DetectorKind, EngineConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    face_core::utils::common::init_logging();

    let config = EngineConfig {
        detector_kind: DetectorKind::Scrfd,
        enable_facemesh: false,
        enable_parsing: false,
        enable_recognition: false,
        enable_attributes: true,
        enable_gaze: true,
        enable_headpose: true,
        enable_quality: false,
        enable_spoofing: false,
        enable_matting: false,
        enable_tracking: false,
        ..Default::default()
    };
    let mut analyzer = FaceAnalyzer::new(config)?;

    // 独立的逐脸预测器 (使用同一 batch 模型, 单条输入)
    let get_bytes = |name: &str| {
        face_core::models::get_model_bytes(name)
            .ok_or_else(|| format!("model {} not embedded", name))
    };
    let em_session =
        InferenceSession::from_bytes("emotion_ferplus", get_bytes("emotion_ferplus")?)?;
    let em_net = EmotionNet::from_session(em_session, EmotionModel::FerPlus)?;
    let gz_session =
        InferenceSession::from_bytes("mobilenetv2_gaze", get_bytes("mobilenetv2_gaze")?)?;
    let gz_net = GazeNet::from_session(gz_session)?;
    let hp_session =
        InferenceSession::from_bytes("resnet18_headpose", get_bytes("resnet18_headpose")?)?;
    let hp_net = HeadPoseNet::from_session(hp_session)?;

    let mut em_net = em_net;
    let mut gz_net = gz_net;
    let mut hp_net = hp_net;

    let images = [
        "uniface-4.1.0/assets/source/detect_group.jpg",
        "uniface-4.1.0/assets/source/detect_crowd.jpg",
        "uniface-4.1.0/assets/source/anon_group.jpg",
        "uniface-4.1.0/assets/source/state_sunglasses.jpg",
        "uniface-4.1.0/assets/source/verify_now_2014.jpg",
    ];

    let mut all_ok = true;
    for path in images {
        let image = face_core::utils::image_utils::load_image_with_exif(path)?;
        let faces = analyzer.analyze(&image)?;
        if faces.len() < 2 {
            println!("{}: {} faces (skip, need >=2)", path, faces.len());
            continue;
        }
        let refs: Vec<&_> = faces.iter().collect();
        println!("{}: {} faces", path, faces.len());

        // emotion: batch vs per-face
        let b = em_net.predict_batch(&image, &refs)?;
        for (i, f) in faces.iter().enumerate() {
            let p = em_net.predict(&image, f)?;
            let bb = b[i].clone().expect("batch emotion none");
            let m = f.emotion.as_ref().unwrap();
            assert_eq!(bb.emotion, p.emotion, "emotion label mismatch");
            let d_batch = (bb.confidence - p.confidence).abs();
            let d_pipe = (f.emotion_confidence.unwrap_or(f32::NAN) - p.confidence).abs();
            let ok = d_batch == 0.0 && d_pipe == 0.0;
            if !ok {
                all_ok = false;
                println!(
                    "  face {}: EMOTION MISMATCH batch={:.6} single={:.6} pipe={:.6}",
                    i,
                    bb.confidence,
                    p.confidence,
                    f.emotion_confidence.unwrap_or(f32::NAN)
                );
            }
        }

        // gaze
        let b = gz_net.predict_batch(&image, &refs)?;
        for (i, f) in faces.iter().enumerate() {
            let p = gz_net.predict(&image, f)?;
            let bb = b[i].clone().expect("batch gaze none");
            let d_yaw = (bb.yaw - p.yaw).abs();
            let d_pitch = (bb.pitch - p.pitch).abs();
            let m = f.gaze.as_ref().unwrap();
            let d_pipe = ((m.yaw - p.yaw).abs() + (m.pitch - p.pitch).abs()) / 2.0;
            let ok = d_yaw == 0.0 && d_pitch == 0.0 && d_pipe == 0.0;
            if !ok {
                all_ok = false;
                println!(
                    "  face {}: GAZE MISMATCH batch=({:.6},{:.6}) single=({:.6},{:.6}) pipe=({:.6},{:.6})",
                    i, bb.yaw, bb.pitch, p.yaw, p.pitch, m.yaw, m.pitch
                );
            }
        }

        // headpose
        let b = hp_net.predict_batch(&image, &refs)?;
        for (i, f) in faces.iter().enumerate() {
            let p = hp_net.predict(&image, f)?;
            let bb = b[i].clone().expect("batch headpose none");
            let m = f.head_pose.as_ref().unwrap();
            let dp = (bb.pitch - p.pitch).abs() + (bb.yaw - p.yaw).abs() + (bb.roll - p.roll).abs();
            let dpipe = (m.pitch - p.pitch).abs() + (m.yaw - p.yaw).abs() + (m.roll - p.roll).abs();
            let ok = dp == 0.0 && dpipe == 0.0;
            if !ok {
                all_ok = false;
                println!(
                    "  face {}: HEADPOSE MISMATCH batch=({:.6},{:.6},{:.6}) single=({:.6},{:.6},{:.6}) pipe=({:.6},{:.6},{:.6})",
                    i, bb.pitch, bb.yaw, bb.roll, p.pitch, p.yaw, p.roll, m.pitch, m.yaw, m.roll
                );
            }
        }
    }
    println!(
        "{}",
        if all_ok {
            "ALL CONSISTENT (diff=0)"
        } else {
            "MISMATCH FOUND"
        }
    );
    Ok(())
}
