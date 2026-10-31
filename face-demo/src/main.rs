// ====================================================================
// face-demo — 人脸分析引擎示例程序集
//
// 本 crate 演示 face-core 库的完整能力, 共包含 15 个独立示例,
// 每个示例对应一类典型的人脸分析任务。
//
// 用法:
//   face-demo <示例名> [输出目录]
//
// 示例列表:
//   01_detection        人脸检测
//   02_alignment        人脸对齐
//   03_verification     人脸验证 (1:1 比对)
//   04_search           人脸搜索 (1:N 匹配)
//   05_analyzer         综合人脸分析
//   06_parsing          人脸解析 (19 类分割)
//   07_anonymization    人脸匿名化 (模糊/像素化)
//   08_gaze             视线估计
//   09_segmentation     人脸分割 (抠像换背景)
//   10_vector_store     人脸特征向量库
//   11_head_pose        头部姿态估计
//   12_recognition      人脸识别 (相似度矩阵)
//   13_matting          人像抠图
//   14_attributes       人脸属性检测
//   15_mesh             稠密人脸网格
// ====================================================================

use anyhow::{Context, Result, bail};
use face_core::inference::{DetectorKind, EngineConfig};
use face_core::pipeline::render;
use face_core::pipeline::stores::{FaceMetadata, FaceStore};
use face_core::types::DetectedFace;
use face_core::{FaceAnalyzer, FaceEngine};
use image::{Rgb, RgbImage};

/// 测试图片目录 (随仓库分发)
const ASSETS: &str = "face-demo/assets";

/// 可视化结果的默认输出根目录
const OUT_DIR: &str = "output/demo";

fn main() -> Result<()> {
    // 初始化日志系统 (按平台自动选择日志后端)
    face_core::utils::common::init_logging();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        // 未指定示例名时打印用法说明
        println!("Usage: face-demo <example-name>");
        println!();
        println!("Examples (mirroring uniface-4.1.0/examples):");
        for name in EXAMPLES {
            println!("  {}", name);
        }
        return Ok(());
    }

    // 第一个参数为示例名, 第二个参数 (可选) 为输出目录
    let name = args[1].as_str();
    let out_dir = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| format!("{}/{}", OUT_DIR, name));

    // 按示例名分派到对应的处理函数
    match name {
        "01_detection" => ex01_detection(&out_dir),
        "02_alignment" => ex02_alignment(&out_dir),
        "03_verification" => ex03_verification(&out_dir),
        "04_search" => ex04_search(&out_dir),
        "05_analyzer" => ex05_analyzer(&out_dir),
        "06_parsing" => ex06_parsing(&out_dir),
        "07_anonymization" => ex07_anonymization(&out_dir),
        "08_gaze" => ex08_gaze(&out_dir),
        "09_segmentation" => ex09_segmentation(&out_dir),
        "10_vector_store" => ex10_vector_store(&out_dir),
        "11_head_pose" => ex11_head_pose(&out_dir),
        "12_recognition" => ex12_recognition(&out_dir),
        "13_matting" => ex13_matting(&out_dir),
        "14_attributes" => ex14_attributes(&out_dir),
        "15_mesh" => ex15_mesh(&out_dir),
        _ => bail!(
            "unknown example '{}'\nrun without arguments to list examples",
            name
        ),
    }
}

/// 全部示例名称, 用于用法提示
const EXAMPLES: [&str; 15] = [
    "01_detection",
    "02_alignment",
    "03_verification",
    "04_search",
    "05_analyzer",
    "06_parsing",
    "07_anonymization",
    "08_gaze",
    "09_segmentation",
    "10_vector_store",
    "11_head_pose",
    "12_recognition",
    "13_matting",
    "14_attributes",
    "15_mesh",
];

// --------------------------------------------------------------------
// 通用辅助函数
// --------------------------------------------------------------------

/// 按路径读取图片, 自动应用 EXIF 方向信息
fn load(path: &str) -> Result<image::DynamicImage> {
    face_core::utils::image_utils::load_image_with_exif(path)
        .with_context(|| format!("could not read {}", path))
}

/// 将图片保存到输出目录 (目录不存在时自动创建)
fn save(img: &image::DynamicImage, out_dir: &str, name: &str) -> Result<()> {
    std::fs::create_dir_all(out_dir)?;
    let path = format!("{}/{}", out_dir, name);
    img.save(&path).with_context(|| format!("save {}", path))?;
    println!("saved {}", path);
    Ok(())
}

/// 构建综合分析引擎: 开启检测、三维网格、特征提取与属性预测,
/// 用于需要多维结果的示例
fn analyzer() -> Result<FaceAnalyzer> {
    let config = EngineConfig {
        detector_kind: DetectorKind::Scrfd,
        enable_facemesh: true,
        enable_recognition: true,
        enable_attributes: true,
        ..Default::default()
    };
    Ok(FaceAnalyzer::new(config)?)
}

/// 构建仅开启检测的轻量引擎, 用于只需要边界框与关键点的示例
fn detector() -> Result<FaceAnalyzer> {
    Ok(FaceAnalyzer::new(EngineConfig {
        detector_kind: DetectorKind::Scrfd,
        ..Default::default()
    })?)
}

/// 从检测结果中提取 5 点关键点 (左眼、右眼、鼻尖、左嘴角、右嘴角),
/// 返回 [(x, y); 5]; 若检测器未输出关键点则返回 None
fn landmarks5(face: &DetectedFace) -> Option<[(f32, f32); 5]> {
    face.landmarks.as_ref().map(|lm| {
        let mut pts = [(0.0f32, 0.0f32); 5];
        for i in 0..5 {
            pts[i] = (lm[[i, 0]], lm[[i, 1]]);
        }
        pts
    })
}

// ====================================================================
// 示例 01 — 人脸检测
//
// 对多人合影执行人脸检测, 打印每张人脸的边界框、置信度和
// 关键点数量, 并输出带标注框的可视化图片。
// ====================================================================
fn ex01_detection(out_dir: &str) -> Result<()> {
    println!("== 01 Face Detection ==");
    let mut det = detector()?;
    let image = load(&format!("{}/detect_group.jpg", ASSETS))?;

    // 执行检测, 返回全部人脸
    let faces = det.analyze(&image)?;
    println!("Detected {} face(s)", faces.len());
    for (i, f) in faces.iter().enumerate() {
        println!(
            "  Face {}: bbox=({:.0},{:.0},{:.0},{:.0}) conf={:.4} landmarks={}",
            i + 1,
            f.bbox.x1,
            f.bbox.y1,
            f.bbox.x2,
            f.bbox.y2,
            f.confidence,
            f.landmarks.as_ref().map(|l| l.nrows()).unwrap_or(0)
        );
    }
    // 绘制边界框 + 关键点后保存
    save(
        &render::draw_faces(&image.to_rgb8(), &faces).into(),
        out_dir,
        "detection.jpg",
    )?;
    Ok(())
}

// ====================================================================
// 示例 02 — 人脸对齐
//
// 基于 5 点关键点计算相似变换矩阵, 把人脸区域矫正并缩放到
// 112x112 标准尺寸 (识别模型的输入规格), 输出对齐后的人脸图。
// ====================================================================
fn ex02_alignment(out_dir: &str) -> Result<()> {
    println!("== 02 Face Alignment ==");
    let mut det = detector()?;
    let names = [
        "age_child.jpg",
        "age_adult.jpg",
        "age_middle.jpg",
        "age_senior.jpg",
        "landmarks_face.jpg",
    ];

    for name in names {
        let image = load(&format!("{}/{}", ASSETS, name))?;
        let faces = det.analyze(&image)?;
        let Some(face) = faces.first() else {
            println!("{}: no face", name);
            continue;
        };
        // 提取 5 点关键点并估计 2x3 前向仿射矩阵
        let pts5 = landmarks5(face)
            .ok_or_else(|| anyhow::anyhow!("{}: detector returned no 5-pt landmarks", name))?;
        let (fwd, _inv) = face_core::utils::alignment::estimate_norm_112(&pts5);
        println!(
            "{}: bbox=({:.0},{:.0},{:.0},{:.0}), fwd=[[{:.4},{:.4},{:.2}],[{:.4},{:.4},{:.2}]]",
            name,
            face.bbox.x1,
            face.bbox.y1,
            face.bbox.x2,
            face.bbox.y2,
            fwd[0][0],
            fwd[0][1],
            fwd[0][2],
            fwd[1][0],
            fwd[1][1],
            fwd[1][2]
        );
        // 用前向矩阵做双线性插值 warp, 得到 112x112 对齐人脸
        let rgb = image.to_rgb8();
        let aligned = face_core::utils::image_utils::warp_affine_bilinear(&rgb, &fwd, 112, 112);
        save(
            &image::DynamicImage::ImageRgb8(aligned),
            out_dir,
            &format!("aligned_{}.jpg", name),
        )?;
    }
    Ok(())
}

// ====================================================================
// 示例 03 — 人脸验证 (1:1 比对)
//
// 分别提取两张图片的人脸特征向量, 计算余弦相似度并与阈值
// 比较, 判断是否为同一人。
// ====================================================================
fn ex03_verification(out_dir: &str) -> Result<()> {
    println!("== 03 Face Verification ==");
    let mut eng = analyzer()?;

    // 待比对的图片对: 前三组为同一人不同年份, 最后一组为不同人
    let pairs = [
        ("verify_now_2010.jpg", "verify_now_2014.jpg"),
        ("verify_now_2014.jpg", "verify_now_2024.jpg"),
        ("verify_now_2010.jpg", "verify_now_2024.jpg"),
        ("verify_now_2024.jpg", "age_adult.jpg"),
    ];
    // 判定阈值: 高于此值视为同一人
    const THRESHOLD: f32 = 0.4;

    for (a, b) in pairs {
        let fa = eng.analyze(&load(&format!("{}/{}", ASSETS, a))?)?;
        let fb = eng.analyze(&load(&format!("{}/{}", ASSETS, b))?)?;
        let (Some(fa), Some(fb)) = (fa.first(), fb.first()) else {
            println!("{} vs {}: no face", a, b);
            continue;
        };
        let (Some(ea), Some(eb)) = (fa.embedding.as_ref(), fb.embedding.as_ref()) else {
            println!("{} vs {}: no embedding", a, b);
            continue;
        };
        // 特征向量已归一化, 点积即为余弦相似度
        let sim = cosine(ea, eb);
        println!(
            "{} vs {}: similarity={:.4} -> {} (threshold={})",
            a,
            b,
            sim,
            if sim > THRESHOLD {
                "Same person"
            } else {
                "Different people"
            },
            THRESHOLD
        );
    }
    Ok(())
}

/// 计算两个向量的余弦相似度 (输入需已归一化, 此处直接求点积)
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

// ====================================================================
// 示例 04 — 人脸搜索 (1:N 匹配)
//
// 在一张多人合影中搜索与查询人脸最相似的人: 对合影中每个人
// 计算与查询特征的相似度, 按分数排序并高亮最佳匹配。
// ====================================================================
fn ex04_search(out_dir: &str) -> Result<()> {
    println!("== 04 Face Search ==");
    let mut eng = analyzer()?;

    // 查询图: 单人正脸; 目标图: 多人合影
    let query = eng.analyze(&load(&format!("{}/verify_einstein_1947.jpg", ASSETS))?)?;
    let group = eng.analyze(&load(&format!("{}/detect_crowd.jpg", ASSETS))?)?;
    println!("Query faces: {}, group faces: {}", query.len(), group.len());

    let q = query[0]
        .embedding
        .as_ref()
        .context("query has no embedding")?;

    // 计算查询向量与合影中每个特征的相似度, 按分数降序排序
    let mut matches: Vec<(usize, f32)> = group
        .iter()
        .enumerate()
        .filter_map(|(i, f)| f.embedding.as_ref().map(|e| (i, cosine(q, e))))
        .collect();
    matches.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());

    println!("Top 5 most similar people:");
    for (rank, (idx, sim)) in matches.iter().take(5).enumerate() {
        println!(
            "  {}. Person #{}: similarity = {:.4}",
            rank + 1,
            idx + 1,
            sim
        );
    }

    // 可视化: 最佳匹配画绿框, 其余人脸画灰框
    let image = load(&format!("{}/detect_crowd.jpg", ASSETS))?;
    let mut canvas = image.to_rgb8();
    let (best_idx, best_sim) = matches[0];
    for (i, f) in group.iter().enumerate() {
        let color = if i == best_idx {
            Rgb([0, 255, 0])
        } else {
            Rgb([128, 128, 128])
        };
        draw_rect(&mut canvas, &f.bbox, color);
    }
    println!(
        "Best match: Person #{} similarity={:.4} (>0.4 => match found)",
        best_idx + 1,
        best_sim
    );
    save(
        &image::DynamicImage::ImageRgb8(canvas),
        out_dir,
        "search.jpg",
    )?;
    Ok(())
}

/// 在画布上绘制 1 像素宽的矩形边框, 自动裁剪到画布范围内
fn draw_rect(canvas: &mut RgbImage, bbox: &face_core::types::BBox, color: Rgb<u8>) {
    let (x1, y1, x2, y2) = bbox.to_int();
    let (x1, y1, x2, y2) = (
        x1.max(0) as u32,
        y1.max(0) as u32,
        (x2.min(canvas.width() as i32 - 1)).max(0) as u32,
        (y2.min(canvas.height() as i32 - 1)).max(0) as u32,
    );
    // 上下两条边
    for x in x1..=x2 {
        canvas.put_pixel(x, y1, color);
        canvas.put_pixel(x, y2, color);
    }
    // 左右两条边
    for y in y1..=y2 {
        canvas.put_pixel(x1, y, color);
        canvas.put_pixel(x2, y, color);
    }
}

// ====================================================================
// 示例 05 — 综合人脸分析
//
// 对多张图片执行检测 + 性别/年龄属性 + 特征提取, 打印每张
// 人脸的分析摘要, 并输出带标注框的可视化图片; 最后计算前
// 两张图片人脸的相似度。
// ====================================================================
fn ex05_analyzer(out_dir: &str) -> Result<()> {
    println!("== 05 Face Analyzer ==");
    let mut eng = analyzer()?;
    let names = ["age_child.jpg", "age_middle.jpg", "detect_group.jpg"];
    // 缓存前两张图的特征向量, 循环结束后做相似度对比
    let mut first_two: Option<Vec<Vec<f32>>> = None;

    for name in names {
        let image = load(&format!("{}/{}", ASSETS, name))?;
        let faces = eng.analyze(&image)?;
        println!("{}: {} face(s)", name, faces.len());
        // 打印每张人脸的性别与年龄属性
        for (i, f) in faces.iter().enumerate() {
            let gender = f
                .gender
                .map(|g| format!("{:?}", g))
                .unwrap_or_else(|| "-".into());
            let age = f
                .age
                .map(|a| format!("{}y", a))
                .unwrap_or_else(|| "-".into());
            println!("  Face {}: {}, {}", i + 1, gender, age);
        }
        if name == names[0] || name == names[1] {
            let e = faces
                .iter()
                .map(|f| f.embedding.clone().unwrap_or_default())
                .collect::<Vec<_>>();
            first_two.get_or_insert(e);
        }
        save(
            &image::DynamicImage::ImageRgb8(render::draw_faces(&image.to_rgb8(), &faces)),
            out_dir,
            &format!("analyzed_{}", name),
        )?;
    }
    // 输出前两张图片人脸的相似度
    if let Some(pairs) = first_two {
        if pairs.len() == 2 && !pairs[0].is_empty() && !pairs[1].is_empty() {
            println!(
                "Similarity between first two images: {:.4}",
                cosine(&pairs[0], &pairs[1])
            );
        }
    }
    Ok(())
}

// ====================================================================
// 示例 06 — 人脸解析 (19 类分割)
//
// 对人脸区域做 19 类语义分割, 输出:
//   1) 遮罩半透明叠加图
//   2) 纯色类别可视化图
//   3) 皮肤 / 头发 / 嘴唇三个部件的单独提取图
// ====================================================================
fn ex06_parsing(out_dir: &str) -> Result<()> {
    println!("== 06 Face Parsing ==");
    let mut engine = FaceEngine::new(EngineConfig::default())?;
    let names = ["parse_face.jpg", "mesh_face.jpg", "age_adult.jpg"];

    for name in names {
        let image = load(&format!("{}/{}", ASSETS, name))?;
        let result = engine.analyze(&image)?;
        let Some(face) = result.faces.iter().find(|f| f.parsing_mask.is_some()) else {
            println!("{}: no parsing mask", name);
            continue;
        };
        let mask = face.parsing_mask.as_ref().unwrap();
        // 统计遮罩中出现的类别
        let unique = unique_labels(mask);
        println!(
            "{}: parsed {} unique classes {:?}",
            name, unique.0, unique.1
        );

        // 解析遮罩覆盖的是按 (0.2, 0.4, 0.2, 0.2) 比例扩展后的
        // 人脸裁剪区域, 因此需要先裁出同尺寸的原图再叠加
        let region = face
            .face
            .bbox
            .expand((0.2, 0.4, 0.2, 0.2))
            .clamp(image.width() as f32, image.height() as f32);
        let (rx1, ry1, _, _) = region.to_int();
        let crop = image
            .crop_imm(rx1 as u32, ry1 as u32, mask.width, mask.height)
            .to_rgb8();
        save(
            &image::DynamicImage::ImageRgb8(render::draw_parsing(&crop, mask)),
            out_dir,
            &format!("parsing_{}", name),
        )?;
        save(
            &image::DynamicImage::ImageRgb8(render::draw_parsing_solid(mask)),
            out_dir,
            &format!("parsing_solid_{}", name),
        )?;

        // 按类别 ID 提取部件: 1=皮肤, 17=头发, 12/13=上下唇
        for (comp, ids) in [
            ("skin", &[1u8][..]),
            ("hair", &[17u8][..]),
            ("lips", &[12u8, 13u8][..]),
        ] {
            let extracted = extract_component(&crop, mask, ids);
            save(
                &image::DynamicImage::ImageRgb8(extracted),
                out_dir,
                &format!("comp_{}_{}", comp, name),
            )?;
        }
    }
    Ok(())
}

/// 统计遮罩中出现的类别 ID 集合, 返回 (类别数, 升序类别列表)
fn unique_labels(mask: &face_core::types::ParsingMask) -> (usize, Vec<u8>) {
    let mut set = std::collections::BTreeSet::new();
    for v in mask.mask.iter() {
        set.insert(*v);
    }
    (set.len(), set.into_iter().collect())
}

/// 按类别 ID 集合提取部件: 命中类别的像素保留原色, 其余置黑
fn extract_component(img: &RgbImage, mask: &face_core::types::ParsingMask, ids: &[u8]) -> RgbImage {
    let mut out = RgbImage::new(img.width(), img.height());
    for y in 0..img.height() {
        for x in 0..img.width() {
            if ids.contains(&mask.mask[[y as usize, x as usize]]) {
                out.put_pixel(x, y, *img.get_pixel(x, y));
            }
        }
    }
    out
}

// ====================================================================
// 示例 07 — 人脸匿名化
//
// 检出人脸后分别用三种方式做隐私处理:
//   盒式模糊 / 像素化 / 强模糊
// ====================================================================
fn ex07_anonymization(out_dir: &str) -> Result<()> {
    println!("== 07 Face Anonymization ==");
    use face_core::pipeline::privacy::{AnonymizeMode, anonymize_regions};
    let mut det = detector()?;
    let image = load(&format!("{}/anon_group.jpg", ASSETS))?;
    let faces = det.analyze(&image)?;
    println!("Detected {} faces", faces.len());
    // 收集全部人脸边界框, 供匿名化函数一次性处理
    let boxes: Vec<_> = faces.iter().map(|f| f.bbox.clone()).collect();

    // 三种匿名化方式: 51 像素核模糊 / 16 像素块像素化 / 强模糊
    let modes: [(String, AnonymizeMode); 3] = [
        ("blur".into(), AnonymizeMode::Blur { kernel: 51 }),
        ("pixelate".into(), AnonymizeMode::Pixelate { block: 16 }),
        ("blackout".into(), AnonymizeMode::BlurStrong),
    ];
    for (name, mode) in modes {
        let out = anonymize_regions(&image, &boxes, &mode);
        println!("method {}: done", name);
        save(&out, out_dir, &format!("anon_{}.jpg", name))?;
    }
    Ok(())
}

// ====================================================================
// 示例 08 — 视线估计
//
// 对视线方向各异的图片估计注视角度, 输出俯仰角 (pitch) 与
// 水平角 (yaw) 的弧度和角度值。
// ====================================================================
fn ex08_gaze(out_dir: &str) -> Result<()> {
    println!("== 08 Gaze Estimation ==");
    let config = EngineConfig {
        detector_kind: DetectorKind::Scrfd,
        enable_gaze: true,
        ..Default::default()
    };
    let mut eng = FaceAnalyzer::new(config)?;
    let names = ["gaze_away.jpg", "gaze_averted.jpg", "gaze_right.jpg"];

    for name in names {
        let image = load(&format!("{}/{}", ASSETS, name))?;
        let faces = eng.analyze(&image)?;
        for (i, f) in faces.iter().enumerate() {
            match &f.gaze {
                Some(g) => println!(
                    "  {} Face {}: pitch={:.4} rad ({:.1}°), yaw={:.4} rad ({:.1}°)",
                    name,
                    i + 1,
                    g.pitch,
                    g.pitch.to_degrees(),
                    g.yaw,
                    g.yaw.to_degrees()
                ),
                None => println!("  {} Face {}: no gaze result", name, i + 1),
            }
        }
        save(
            &image::DynamicImage::ImageRgb8(render::draw_faces(&image.to_rgb8(), &faces)),
            out_dir,
            &format!("gaze_{}", name),
        )?;
    }
    Ok(())
}

// ====================================================================
// 示例 09 — 人脸分割 (抠像换背景)
//
// 利用解析遮罩把人脸前景从原图中分离, 遮罩区域外的背景替换
// 为纯白色, 输出换背景结果图。
// ====================================================================
fn ex09_segmentation(out_dir: &str) -> Result<()> {
    println!("== 09 Face Segmentation ==");
    let mut engine = FaceEngine::new(EngineConfig::default())?;
    let image = load(&format!("{}/seg_face.jpg", ASSETS))?;
    let result = engine.analyze(&image)?;

    let Some(face) = result.faces.iter().find(|f| f.parsing_mask.is_some()) else {
        bail!("no face / parsing mask in seg_face.jpg");
    };
    let mask = face.parsing_mask.as_ref().unwrap();
    let rgb = image.to_rgb8();

    // 前景类别集合: 皮肤/眉毛/眼睛/鼻子/嘴唇等面部区域
    // 遮罩对应按 (0.2, 0.4, 0.2, 0.2) 扩展的人脸裁剪区域
    let region = face
        .face
        .bbox
        .expand((0.2, 0.4, 0.2, 0.2))
        .clamp(rgb.width() as f32, rgb.height() as f32);
    let (rx1, ry1, _, _) = region.to_int();
    // 先整体填充白色作为新背景
    let mut out = RgbImage::from_pixel(rgb.width(), rgb.height(), Rgb([255, 255, 255]));
    let fg = [1u8, 2, 3, 4, 5, 10, 11, 12, 13, 17];
    // 遮罩区域内: 命中前景类别的像素保留原图颜色
    for y in 0..rgb.height() {
        for x in 0..rgb.width() {
            let in_region = x >= rx1 as u32
                && y >= ry1 as u32
                && x < rx1 as u32 + mask.width
                && y < ry1 as u32 + mask.height;
            if in_region {
                let mx = (x - rx1 as u32) as usize;
                let my = (y - ry1 as u32) as usize;
                if fg.contains(&mask.mask[[my, mx]]) {
                    out.put_pixel(x, y, *rgb.get_pixel(x, y));
                }
            }
        }
    }
    // 遮罩区域外: 恢复为原图像素, 保证只影响人脸附近区域
    image::imageops::overlay(
        &mut out,
        &rgb_cropped_marker(&rgb, rx1, ry1, mask.width, mask.height),
        0,
        0,
    );
    save(
        &image::DynamicImage::ImageRgb8(out),
        out_dir,
        "segmentation_white_bg.jpg",
    )?;
    println!("face extracted on white background");
    Ok(())
}

/// 生成一张只保留"指定区域之外"像素的图像, 用于把区域外的
/// 原图像素拷回结果图
fn rgb_cropped_marker(rgb: &RgbImage, rx: i32, ry: i32, mw: u32, mh: u32) -> RgbImage {
    let mut out = RgbImage::new(rgb.width(), rgb.height());
    for y in 0..rgb.height() {
        for x in 0..rgb.width() {
            let in_region =
                x >= rx as u32 && y >= ry as u32 && x < rx as u32 + mw && y < ry as u32 + mh;
            if !in_region {
                out.put_pixel(x, y, *rgb.get_pixel(x, y));
            }
        }
    }
    out
}

// ====================================================================
// 示例 10 — 人脸特征向量库
//
// 演示特征库的完整工作流: 入库 (注册多人特征) -> 相似检索
// (取前 3) -> 阈值匹配 (返回首个过阈值的结果)。
// ====================================================================
fn ex10_vector_store(_out_dir: &str) -> Result<()> {
    println!("== 10 Face Vector Store ==");
    let mut eng = analyzer()?;
    let mut store = FaceStore::new();

    // 注册 5 个人, 每人取首张人脸的特征向量入库
    let enrolled = [
        ("verify_now_2010.jpg", "person_a"),
        ("age_child.jpg", "person_b"),
        ("age_adult.jpg", "person_c"),
        ("age_middle.jpg", "person_d"),
        ("age_senior.jpg", "person_e"),
    ];
    for (path, label) in enrolled {
        let faces = eng.analyze(&load(&format!("{}/{}", ASSETS, path))?)?;
        let Some(f) = faces.first() else { continue };
        let Some(emb) = f.embedding.as_ref() else {
            continue;
        };
        let idx = store.add(
            emb.clone(),
            FaceMetadata {
                name: Some(label.into()),
                gender: None,
                age: None,
                emotion: None,
                added_at: 0,
            },
        );
        println!(
            "enrolled {} as store index {} ({}-D)",
            label,
            idx,
            emb.len()
        );
    }
    println!("store size: {} vectors", store.len());

    // 用一张新图的特征做查询
    let query = eng.analyze(&load(&format!("{}/verify_now_2024.jpg", ASSETS))?)?;
    let q = query[0]
        .embedding
        .as_ref()
        .context("query has no embedding")?;
    // 相似检索: 返回相似度最高的前 3 条
    let top = store.search(q, 3);
    println!("top-3 for verify_now_2024.jpg:");
    for r in &top {
        println!(
            "  index={} label={:?} score={:.4}",
            r.index,
            r.metadata.name.as_deref().unwrap_or("-"),
            r.score
        );
    }
    // 阈值匹配: 相似度 >= 0.4 时判定为命中
    let best = store.find_best_match(q, 0.4);
    match best {
        Some(r) => println!(
            "best match above 0.4: index={} label={:?} score={:.4}",
            r.index,
            r.metadata.name.as_deref().unwrap_or("-"),
            r.score
        ),
        None => println!("no match above threshold"),
    }
    Ok(())
}

// ====================================================================
// 示例 11 — 头部姿态估计
//
// 对朝向不同的人脸图片估计三维头部姿态, 输出俯仰 (pitch)、
// 偏航 (yaw)、翻滚 (roll) 三个欧拉角 (弧度)。
// ====================================================================
fn ex11_head_pose(out_dir: &str) -> Result<()> {
    println!("== 11 Head Pose ==");
    let config = EngineConfig {
        detector_kind: DetectorKind::Scrfd,
        enable_headpose: true,
        ..Default::default()
    };
    let mut eng = FaceAnalyzer::new(config)?;
    let names = ["pose_center.jpg", "pose_left.jpg", "pose_right.jpg"];

    for name in names {
        let image = load(&format!("{}/{}", ASSETS, name))?;
        let faces = eng.analyze(&image)?;
        for (i, f) in faces.iter().enumerate() {
            match &f.head_pose {
                Some(p) => println!(
                    "  {} Face {}: pitch={:.3} yaw={:.3} roll={:.3} rad",
                    name,
                    i + 1,
                    p.pitch,
                    p.yaw,
                    p.roll
                ),
                None => println!("  {} Face {}: no head pose", name, i + 1),
            }
        }
        save(
            &image::DynamicImage::ImageRgb8(render::draw_faces(&image.to_rgb8(), &faces)),
            out_dir,
            &format!("pose_{}", name),
        )?;
    }
    Ok(())
}

// ====================================================================
// 示例 12 — 人脸识别 (相似度矩阵)
//
// 提取多张人脸的特征向量, 输出两两之间的余弦相似度矩阵,
// 用于观察同一人不同照片与不同人之间的分数差异。
// ====================================================================
fn ex12_recognition(_out_dir: &str) -> Result<()> {
    println!("== 12 Face Recognition ==");
    let mut eng = analyzer()?;
    // 前 3 张为同一人的不同年份照片, 第 4 张为另一人
    let subjects = [
        ("a_2010", "verify_now_2010.jpg"),
        ("a_2014", "verify_now_2014.jpg"),
        ("a_2024", "verify_now_2024.jpg"),
        ("b_child", "age_child.jpg"),
    ];

    let mut embeddings: Vec<(&str, Vec<f32>)> = Vec::new();
    for (label, path) in subjects {
        let faces = eng.analyze(&load(&format!("{}/{}", ASSETS, path))?)?;
        if let Some(e) = faces.first().and_then(|f| f.embedding.clone()) {
            embeddings.push((label, e));
        }
    }

    // 打印 N x N 相似度矩阵: 对角线恒为 1, 同人互比远高于异人
    println!("pairwise cosine similarity:");
    print!("          ");
    for (l, _) in &embeddings {
        print!("{:>9}", l);
    }
    println!();
    for (la, ea) in &embeddings {
        print!("{:>9} ", la);
        for (_, eb) in &embeddings {
            print!("{:9.4}", cosine(ea, eb));
        }
        println!();
    }
    Ok(())
}

// ====================================================================
// 示例 13 — 人像抠图
//
// 对整图做人像抠图, 输出灰度 alpha 遮罩, 并以该遮罩把人像
// 合成到纯白背景上。
// ====================================================================
fn ex13_matting(out_dir: &str) -> Result<()> {
    println!("== 13 Portrait Matting ==");
    let config = EngineConfig {
        detector_kind: DetectorKind::Scrfd,
        enable_matting: true,
        ..Default::default()
    };
    let mut eng = FaceAnalyzer::new(config)?;
    let image = load(&format!("{}/matte_face.jpg", ASSETS))?;
    let matte = eng.matting(&image)?;
    println!(
        "Matte: {}x{}, range [{:.3}, {:.3}]",
        matte.width,
        matte.height,
        matte_min(&matte),
        matte_max(&matte)
    );

    // alpha 遮罩按 0-255 灰度输出为 PNG
    let mut alpha = image::GrayImage::new(matte.width, matte.height);
    for y in 0..matte.height {
        for x in 0..matte.width {
            let v = (matte.alpha[[y as usize, x as usize]] * 255.0).clamp(0.0, 255.0) as u8;
            alpha.put_pixel(x, y, image::Luma([v]));
        }
    }
    save(
        &image::DynamicImage::ImageLuma8(alpha),
        out_dir,
        "matte_alpha.png",
    )?;

    // 按 alpha 逐像素混合: 前景权重 a, 背景权重 1-a (背景为纯白)
    let rgb = image.to_rgb8();
    let mut comp = RgbImage::new(rgb.width(), rgb.height());
    for y in 0..rgb.height() {
        for x in 0..rgb.width() {
            let a = matte.alpha[[y as usize, x as usize]];
            let p = rgb.get_pixel(x, y);
            comp.put_pixel(
                x,
                y,
                Rgb([
                    (p[0] as f32 * a + 255.0 * (1.0 - a)) as u8,
                    (p[1] as f32 * a + 255.0 * (1.0 - a)) as u8,
                    (p[2] as f32 * a + 255.0 * (1.0 - a)) as u8,
                ]),
            );
        }
    }
    save(
        &image::DynamicImage::ImageRgb8(comp),
        out_dir,
        "matte_white_bg.jpg",
    )?;
    Ok(())
}

/// 遮罩最小值, 用于观察 alpha 的动态范围
fn matte_min(m: &face_core::types::MatteResult) -> f32 {
    m.alpha.iter().copied().fold(f32::INFINITY, f32::min)
}

/// 遮罩最大值
fn matte_max(m: &face_core::types::MatteResult) -> f32 {
    m.alpha.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

// ====================================================================
// 示例 14 — 人脸属性检测
//
// 对每组测试图预测 5 项独立属性概率: 左右眼睁开、眼镜、
// 口罩、墨镜, 并与图集的预期状态对照。
// ====================================================================
fn ex14_attributes(out_dir: &str) -> Result<()> {
    println!("== 14 Face Attributes ==");
    let config = EngineConfig {
        detector_kind: DetectorKind::Scrfd,
        enable_attributes: true,
        ..Default::default()
    };
    let mut eng = FaceAnalyzer::new(config)?;
    // (图片, 预期属性) 对照表
    let subjects = [
        ("age_adult.jpg", "eyes open"),
        ("mesh_face.jpg", "no accessories"),
        ("state_b_glasses.jpg", "glasses"),
        ("state_b_sunglasses.jpg", "sunglasses"),
        ("state_b_mask.jpg", "mask"),
    ];

    for (name, expect) in subjects {
        let image = load(&format!("{}/{}", ASSETS, name))?;
        let faces = eng.analyze(&image)?;
        let Some(f) = faces.first() else {
            println!("{}: no face", name);
            continue;
        };
        match &f.face_state {
            Some(s) => println!(
                "  {} (expected: {}): l_eye={:.3} r_eye={:.3} glasses={:.3} mask={:.3} sunglasses={:.3}",
                name, expect, s.left_eye_open, s.right_eye_open, s.eyeglasses, s.mask, s.sunglasses
            ),
            None => println!("  {} (expected: {}): no face_state", name, expect),
        }
        save(
            &image::DynamicImage::ImageRgb8(render::draw_faces(&image.to_rgb8(), &faces)),
            out_dir,
            &format!("attrs_{}", name),
        )?;
    }
    Ok(())
}

// ====================================================================
// 示例 15 — 稠密人脸网格
//
// 检测人脸并预测 468 点三维稠密网格, 输出网格点数与置信度,
// 并保存带网格点标注的可视化图片。
// ====================================================================
fn ex15_mesh(out_dir: &str) -> Result<()> {
    println!("== 15 Face Mesh ==");
    let config = EngineConfig {
        detector_kind: DetectorKind::Scrfd,
        enable_facemesh: true,
        ..Default::default()
    };
    let mut eng = FaceAnalyzer::new(config)?;
    let image = load(&format!("{}/mesh_face.jpg", ASSETS))?;
    let faces = eng.analyze(&image)?;
    println!("Detected {} face(s)", faces.len());
    // 打印每张人脸的网格点数与存在置信度
    for (i, f) in faces.iter().enumerate() {
        if let Some(score) = f.mesh_score {
            let n = f.mesh_3d.as_ref().map(|m| m.nrows()).unwrap_or(0);
            println!("  Face {}: mesh {} points, score={:.4}", i + 1, n, score);
        }
    }
    save(
        &image::DynamicImage::ImageRgb8(render::draw_faces(&image.to_rgb8(), &faces)),
        out_dir,
        "mesh.jpg",
    )?;
    Ok(())
}
