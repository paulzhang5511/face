// ====================================================================
// face-bench — face-core 性能测试工具
//
// 分阶段计时: 检测 / FaceMesh / 识别 / 年龄性别 / 情绪 / 人脸状态 /
//             视线 / 头姿 / 质量 / 活体
// 用法:
//   face-bench <目录或图片> [--rounds N] [--warmup N] [--out 报告.md] [--json 结果.json]
// ====================================================================

use anyhow::{Result, bail};
use face_core::prelude::{DetectorKind, EngineConfig};
use face_core::utils::image_utils::load_image_with_exif;
use face_core::{FaceAnalyzer, StageTimings};
use image::GenericImageView;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// 每阶段的计时样本 (毫秒)
#[derive(Default, Clone)]
struct Samples {
    data: Vec<f64>,
}

impl Samples {
    fn push(&mut self, ms: f64) {
        self.data.push(ms);
    }
    fn mean(&self) -> f64 {
        if self.data.is_empty() {
            0.0
        } else {
            self.data.iter().sum::<f64>() / self.data.len() as f64
        }
    }
    fn min(&self) -> f64 {
        self.data.iter().cloned().fold(f64::INFINITY, f64::min)
    }
    fn max(&self) -> f64 {
        self.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }
    fn std(&self) -> f64 {
        if self.data.len() < 2 {
            return 0.0;
        }
        let m = self.mean();
        (self.data.iter().map(|t| (t - m).powi(2)).sum::<f64>() / (self.data.len() - 1) as f64)
            .sqrt()
    }
    fn n(&self) -> usize {
        self.data.len()
    }
}

/// 每阶段统计 (跨轮聚合)
#[derive(Default)]
struct StageStat {
    samples: Samples,
}

/// 单图结果
struct ImageResult {
    name: String,
    faces: usize,
    width: u32,
    height: u32,
    total: Samples,
    stages: Vec<(&'static str, Samples)>,
}

const STAGES: [&str; 10] = [
    "detect",
    "facemesh",
    "recognition",
    "age_gender",
    "emotion",
    "face_attrib",
    "gaze",
    "headpose",
    "quality",
    "spoofing",
];

fn main() -> Result<()> {
    face_core::utils::common::init_logging();
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        println!(
            "Usage: face-bench <目录或图片> [--rounds N] [--warmup N] [--out report.md] [--json result.json]"
        );
        bail!("missing path argument");
    }
    let path = PathBuf::from(&args[1]);
    let mut rounds = 10usize;
    let mut warmup = 2usize;
    let mut out_md = PathBuf::from("perf_report.md");
    let mut out_json: Option<PathBuf> = None;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--rounds" => {
                rounds = args[i + 1].parse()?;
                i += 2;
            }
            "--warmup" => {
                warmup = args[i + 1].parse()?;
                i += 2;
            }
            "--out" => {
                out_md = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--json" => {
                out_json = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            other => bail!("unknown argument: {}", other),
        }
    }

    let images = collect_images(&path)?;

    // 与 --bench 一致的全 predictor 配置
    let config = EngineConfig {
        detector_kind: DetectorKind::Scrfd,
        enable_facemesh: true,
        enable_parsing: false,
        enable_recognition: true,
        enable_attributes: true,
        enable_gaze: true,
        enable_headpose: true,
        enable_quality: true,
        enable_spoofing: true,
        enable_matting: false,
        enable_tracking: false,
        ..Default::default()
    };
    let mut analyzer = FaceAnalyzer::new(config)?;

    let mut results: Vec<ImageResult> = Vec::new();
    let t_wall = Instant::now();
    for img_path in &images {
        let image = load_image_with_exif(&img_path.to_string_lossy())?;
        let (w, h) = image.dimensions();
        let name = img_path.file_name().unwrap().to_string_lossy().to_string();

        // warmup
        let mut nfaces = 0;
        for _ in 0..warmup {
            nfaces = analyzer.analyze(&image)?.len();
        }
        eprintln!("warmup {}: {} faces", name, nfaces);

        let mut stages: Vec<(&'static str, Samples)> =
            STAGES.iter().map(|&s| (s, Samples::default())).collect();
        let mut total = Samples::default();

        for _ in 0..rounds {
            let t0 = Instant::now();
            let (_faces, t) = analyzer.analyze_profiled(&image)?;
            total.push(t0.elapsed().as_secs_f64() * 1000.0);
            for (slot, (name, ms)) in stages.iter_mut().zip(t.iter()) {
                debug_assert_eq!(slot.0, name);
                slot.1.push(ms);
            }
        }

        results.push(ImageResult {
            name,
            faces: nfaces,
            width: w,
            height: h,
            total,
            stages,
        });
    }
    let wall = t_wall.elapsed().as_secs_f64();

    print_console(&results, rounds, warmup, wall);
    write_markdown(&out_md, &results, rounds, warmup, wall)?;
    println!("\nreport: {}", out_md.display());
    if let Some(p) = out_json {
        write_json(&p, &results, rounds, warmup)?;
        println!("json:    {}", p.display());
    }
    Ok(())
}

fn collect_images(path: &Path) -> Result<Vec<PathBuf>> {
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }
    if !path.is_dir() {
        bail!("path not found: {}", path.display());
    }
    let mut out: Vec<PathBuf> = std::fs::read_dir(path)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            matches!(
                p.extension()
                    .and_then(|e| e.to_str())
                    .map(|s| s.to_lowercase())
                    .as_deref(),
                Some("jpg" | "jpeg" | "png" | "bmp" | "webp")
            )
        })
        .collect();
    out.sort();
    Ok(out)
}

fn print_console(results: &[ImageResult], rounds: usize, warmup: usize, wall: f64) {
    println!(
        "\n=== face-bench: {} image(s) x {} rounds (warmup {}) | wall {:.1}s ===\n",
        results.len(),
        rounds,
        warmup,
        wall
    );
    println!(
        "{:<26}{:>6}{:>9}{:>9}{:>9}{:>8}",
        "image", "faces", "mean(ms)", "min", "max", "std"
    );
    println!("{}", "-".repeat(67));
    for r in results {
        println!(
            "{:<26}{:>6}{:>9.1}{:>9.1}{:>9.1}{:>8.1}",
            r.name,
            r.faces,
            r.total.mean(),
            r.total.min(),
            r.total.max(),
            r.total.std()
        );
    }

    // 阶段汇总 (跨图累计)
    println!("\n--- 阶段耗时 (跨图累计) ---\n");
    println!(
        "{:<14}{:>9}{:>9}{:>9}{:>8}{:>8}",
        "stage", "sum(ms)", "mean", "min", "max", "share%"
    );
    println!("{}", "-".repeat(60));
    let mut stage_sums: Vec<(&'static str, f64)> = Vec::new();
    let mut grand = 0.0;
    for (si, &stage) in STAGES.iter().enumerate() {
        let sum: f64 = results.iter().map(|r| r.stages[si].1.mean()).sum();
        stage_sums.push((stage, sum));
        grand += sum;
    }
    for (stage, sum) in &stage_sums {
        let mins: Vec<f64> = results
            .iter()
            .map(|r| {
                r.stages[STAGES.iter().position(|s| s == stage).unwrap()]
                    .1
                    .min()
            })
            .collect();
        let maxs: Vec<f64> = results
            .iter()
            .map(|r| {
                r.stages[STAGES.iter().position(|s| s == stage).unwrap()]
                    .1
                    .max()
            })
            .collect();
        println!(
            "{:<14}{:>9.1}{:>9.1}{:>9.1}{:>8.1}{:>8.1}",
            stage,
            sum,
            sum / results.len() as f64,
            mins.iter().cloned().fold(f64::INFINITY, f64::min),
            maxs.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            sum / grand * 100.0
        );
    }
    println!(
        "{:<14}{:>9.1}{:>9.1}",
        "TOTAL",
        grand,
        grand / results.len() as f64
    );
}

fn write_markdown(
    path: &Path,
    results: &[ImageResult],
    rounds: usize,
    warmup: usize,
    wall: f64,
) -> Result<()> {
    let mut s = String::new();
    s.push_str("# face-core 性能测试报告\n\n");
    s.push_str(&format!("- 生成时间: {}\n", now_string()));
    s.push_str(&format!(
        "- 图片数: {} | 每图轮数: {} | warmup: {}\n",
        results.len(),
        rounds,
        warmup
    ));
    s.push_str(&format!("- 总墙钟: {:.1}s\n", wall));
    s.push_str("- 配置: SCRFD + FaceMesh + ArcFace + AgeGender + FER+ + FaceAttribNet + Gaze + HeadPose + eDifFIQA + MiniFASNet (全部 batch 路径)\n");
    s.push_str("- 计时: analyze_profiled 分阶段统计, release 构建\n\n");

    // 总耗时表
    s.push_str("## 1. 端到端耗时 (每图 mean of N 轮)\n\n");
    s.push_str("| image | faces | size | mean(ms) | min | max | std |\n|---|---:|---|---:|---:|---:|---:|\n");
    for r in results {
        s.push_str(&format!(
            "| {} | {} | {}x{} | {:.1} | {:.1} | {:.1} | {:.1} |\n",
            r.name,
            r.faces,
            r.width,
            r.height,
            r.total.mean(),
            r.total.min(),
            r.total.max(),
            r.total.std()
        ));
    }
    let total_faces: usize = results.iter().map(|r| r.faces).sum();
    let weighted: f64 = {
        let n: f64 = results.iter().map(|r| r.total.n() as f64).sum();
        if n > 0.0 {
            results
                .iter()
                .map(|r| r.total.mean() * r.total.n() as f64)
                .sum::<f64>()
                / n
        } else {
            0.0
        }
    };
    let total_mean: f64 = results.iter().map(|r| r.total.mean()).sum::<f64>();
    s.push_str(&format!(
        "| **SUM ({} 图 / {} 脸)** | **{}** |  | **{:.1}** |  |  |  |\n",
        results.len(),
        total_faces,
        total_faces,
        total_mean
    ));
    s.push_str(&format!(
        "| **加权平均/图** |  |  | **{:.1}** |  |  |  |\n",
        weighted
    ));

    // 阶段耗时表
    s.push_str("\n## 2. 分阶段耗时 (跨图累计, ms/轮)\n\n");
    s.push_str("| stage | sum | mean/图 | share% |\n|---|---:|---:|---:|\n");
    let mut stage_rows: Vec<(&'static str, f64)> = Vec::new();
    for (si, &stage) in STAGES.iter().enumerate() {
        stage_rows.push((stage, results.iter().map(|r| r.stages[si].1.mean()).sum()));
    }
    let grand: f64 = stage_rows.iter().map(|r| r.1).sum();
    for (stage, sum) in &stage_rows {
        s.push_str(&format!(
            "| {} | {:.1} | {:.1} | {:.1}% |\n",
            stage,
            sum,
            sum / results.len() as f64,
            sum / grand * 100.0
        ));
    }
    s.push_str(&format!(
        "| **TOTAL** | **{:.1}** | **{:.1}** | **100%** |\n",
        grand,
        grand / results.len() as f64
    ));

    // 每图分阶段明细 (Top 耗时图)
    s.push_str("\n## 3. 每图分阶段明细 (多脸图)\n\n");
    let multi: Vec<&ImageResult> = results.iter().filter(|r| r.faces > 1).collect();
    if !multi.is_empty() {
        let mut header = String::from("| image | faces |");
        for &st in STAGES.iter() {
            header.push_str(&format!(" {} |", st));
        }
        header.push_str(" total |");
        s.push_str(&header);
        s.push_str("\n|---|---:|");
        for _ in STAGES.iter() {
            s.push_str("---:|");
        }
        s.push_str("---:|\n");
        for r in &multi {
            let mut row = format!("| {} | {} |", r.name, r.faces);
            for st in &r.stages {
                row.push_str(&format!(" {:.0} |", st.1.mean()));
            }
            row.push_str(&format!(" {:.0} |", r.total.mean()));
            s.push_str(&row);
            s.push('\n');
        }
    }

    // 吞吐
    s.push_str(&format!(
        "\n## 4. 吞吐\n\n- 单脸图 (1 face): 平均 {:.0} ms/图, 即 {:.1} FPS\n- 多脸图: 平均 {:.0} ms/图\n- 全集 46 图 / {} 脸: {:.1} ms, 人均 {:.1} ms/脸\n",
        results.iter().filter(|r| r.faces == 1).map(|r| r.total.mean()).sum::<f64>()
            / results.iter().filter(|r| r.faces == 1).count().max(1) as f64,
        1000.0 / results.iter().filter(|r| r.faces == 1).map(|r| r.total.mean()).sum::<f64>()
            .max(f64::EPSILON) * results.iter().filter(|r| r.faces == 1).count().max(1) as f64,
        {
            let m: Vec<&ImageResult> = results.iter().filter(|r| r.faces > 1).collect();
            if m.is_empty() { 0.0 } else { m.iter().map(|r| r.total.mean()).sum::<f64>() / m.len() as f64 }
        },
        total_faces,
        total_mean,
        total_mean / total_faces.max(1) as f64,
    ));

    std::fs::write(path, s)?;
    Ok(())
}

fn write_json(path: &Path, results: &[ImageResult], rounds: usize, warmup: usize) -> Result<()> {
    let mut imgs = Vec::new();
    for r in results {
        let mut stages = serde_json::Map::new();
        for st in &r.stages {
            stages.insert(
                st.0.to_string(),
                serde_json::json!({
                    "mean": st.1.mean(), "min": st.1.min(), "max": st.1.max(), "std": st.1.std()
                }),
            );
        }
        imgs.push(serde_json::json!({
            "image": r.name,
            "faces": r.faces,
            "size": [r.width, r.height],
            "total_ms": {"mean": r.total.mean(), "min": r.total.min(), "max": r.total.max(), "std": r.total.std()},
            "stages": stages,
        }));
    }
    let root = serde_json::json!({
        "tool": "face-bench",
        "rounds": rounds,
        "warmup": warmup,
        "images": imgs,
    });
    std::fs::write(path, serde_json::to_string_pretty(&root)?)?;
    Ok(())
}

fn now_string() -> String {
    // 无 chrono 依赖: 用系统时间戳
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("unix-{}", d.as_secs())
}
