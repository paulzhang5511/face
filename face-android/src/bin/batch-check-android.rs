// Android 精度验证工具: batch vs 逐脸推理一致性
use anyhow::Result;
use face_core::FaceAnalyzer;
use face_core::inference::InferenceSession;
use face_core::pipeline::attribute::{EmotionModel, EmotionNet};
use face_core::pipeline::gaze::GazeNet;
use face_core::pipeline::headpose::HeadPoseNet;
use face_core::prelude::{DetectorKind, EngineConfig};

fn main() -> Result<()> {
    face_android::init_logging();

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

    let get_bytes = |name: &str| -> Result<&'static [u8]> {
        Ok(face_core::models::get_model_bytes(name)
            .ok_or_else(|| anyhow::anyhow!("model {} not embedded", name))?)
    };
    let em_net = EmotionNet::from_session(
        InferenceSession::from_bytes("emotion_ferplus", get_bytes("emotion_ferplus")?)?,
        EmotionModel::FerPlus,
    )?;
    let gz_net = GazeNet::from_session(InferenceSession::from_bytes(
        "mobilenetv2_gaze",
        get_bytes("mobilenetv2_gaze")?,
    )?)?;
    let hp_net = HeadPoseNet::from_session(InferenceSession::from_bytes(
        "resnet18_headpose",
        get_bytes("resnet18_headpose")?,
    )?)?;
    let mut em_net = em_net;
    let mut gz_net = gz_net;
    let mut hp_net = hp_net;

    let args: Vec<String> = std::env::args().collect();
    let base = if args.len() > 1 {
        args[1].clone()
    } else {
        ".".to_string()
    };
    let images = [
        "detect_group.jpg",
        "detect_crowd.jpg",
        "anon_group.jpg",
        "state_sunglasses.jpg",
        "verify_now_2014.jpg",
    ];

    let mut all_ok = true;
    for name in images {
        let path = format!("{}/{}", base, name);
        let image = match face_core::utils::image_utils::load_image_with_exif(&path) {
            Ok(im) => im,
            Err(e) => {
                println!("{}: load failed: {}", path, e);
                continue;
            }
        };
        let faces = analyzer.analyze(&image)?;
        if faces.len() < 2 {
            println!("{}: {} faces (skip)", path, faces.len());
            continue;
        }
        let refs: Vec<&_> = faces.iter().collect();
        println!("{}: {} faces", path, faces.len());

        let b = em_net.predict_batch(&image, &refs)?;
        for (i, f) in faces.iter().enumerate() {
            let p = em_net.predict(&image, f)?;
            let bb = b[i].clone().expect("batch emotion none");
            let m = f.emotion.as_ref().unwrap();
            let d_pipe = (f.emotion_confidence.unwrap_or(f32::NAN) - p.confidence).abs();
            let d_bs = (bb.confidence - p.confidence).abs();
            let ok = bb.emotion == p.emotion && d_bs <= 1e-5 && d_pipe <= 1e-5;
            if !ok {
                all_ok = false;
                println!(
                    "  face {}: EMOTION MISMATCH batch={:.6} single={:.6} d_bs={:.3e} d_pipe={:.3e}",
                    i, bb.confidence, p.confidence, d_bs, d_pipe
                );
            }
        }

        let b = gz_net.predict_batch(&image, &refs)?;
        for (i, f) in faces.iter().enumerate() {
            let p = gz_net.predict(&image, f)?;
            let bb = b[i].clone().expect("batch gaze none");
            let m = f.gaze.as_ref().unwrap();
            let d_bs = ((bb.yaw - p.yaw).abs()).max((bb.pitch - p.pitch).abs());
            let d_pipe = ((m.yaw - p.yaw).abs()).max((m.pitch - p.pitch).abs());
            let ok = d_bs <= 1e-3 && d_pipe <= 1e-3;
            if !ok {
                all_ok = false;
                println!(
                    "  face {}: GAZE MISMATCH d_bs={:.3e} d_pipe={:.3e} (batch=({:.4},{:.4}) single=({:.4},{:.4}))",
                    i, d_bs, d_pipe, bb.yaw, bb.pitch, p.yaw, p.pitch
                );
            }
        }

        let b = hp_net.predict_batch(&image, &refs)?;
        for (i, f) in faces.iter().enumerate() {
            let p = hp_net.predict(&image, f)?;
            let bb = b[i].clone().expect("batch headpose none");
            let m = f.head_pose.as_ref().unwrap();
            let d_bs = ((bb.pitch - p.pitch).abs())
                .max((bb.yaw - p.yaw).abs())
                .max((bb.roll - p.roll).abs());
            let d_pipe = ((m.pitch - p.pitch).abs())
                .max((m.yaw - p.yaw).abs())
                .max((m.roll - p.roll).abs());
            let ok = d_bs <= 1e-3 && d_pipe <= 1e-3;
            if !ok {
                all_ok = false;
                println!(
                    "  face {}: HEADPOSE MISMATCH d_bs={:.3e} d_pipe={:.3e}",
                    i, d_bs, d_pipe
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
