// 10次基准对比: 与 uniface FaceAnalyzer 同管线 (SCRFD + ArcFace + AgeGender + FaceAttribNet)
use face_core::FaceAnalyzer;
use face_core::prelude::{DetectorKind, EngineConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    face_core::utils::common::init_logging();
    let config = EngineConfig {
        detector_kind: DetectorKind::Scrfd,
        enable_facemesh: false,
        enable_parsing: false,
        enable_recognition: true,
        enable_attributes: true,
        enable_gaze: false,
        enable_headpose: false,
        enable_quality: false,
        enable_spoofing: false,
        enable_matting: false,
        enable_tracking: false,
        ..Default::default()
    };
    let mut analyzer = FaceAnalyzer::new(config)?;

    let images = ["detect_group.jpg", "detect_crowd.jpg"];
    let n = 10;
    let mut results: Vec<(String, Vec<f64>, usize)> = Vec::new();

    for name in images {
        let path = format!("face-demo/assets/{}", name);
        let image = face_core::utils::image_utils::load_image_with_exif(&path)?;
        // warmup
        let faces = analyzer.analyze(&image)?;
        let nf = faces.len();
        let mut times = Vec::with_capacity(n);
        for _ in 0..n {
            let t0 = std::time::Instant::now();
            analyzer.analyze(&image)?;
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        results.push((name.to_string(), times, nf));
    }

    println!(
        "{:<18}{:>6}{:>13}{:>9}{:>9}{:>9}",
        "image", "faces", "rs mean(ms)", "rs min", "rs max", "rs std"
    );
    for (name, times, nf) in &results {
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let var = times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / (times.len() - 1) as f64;
        let std = var.sqrt();
        println!(
            "{:<18}{:>6}{:>13.1}{:>9.1}{:>9.1}{:>9.1}",
            name, nf, mean, min, max, std
        );
    }
    Ok(())
}
