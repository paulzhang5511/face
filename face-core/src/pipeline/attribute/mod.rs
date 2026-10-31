// ====================================================================
// 人脸属性预测 — 年龄/性别、情绪、人脸状态
// ====================================================================
//
// 属性预测统一实现 (年龄/性别 / 情绪 / 人脸状态)
//
// 公共流程: 从 face.bbox / face.landmarks 裁脸
// → resize → ONNX 推理 → 后处理为类型化结果

use anyhow::Result;
use image::{DynamicImage, GenericImageView};
use log::info;
use ndarray::Array4;
use ort::value::TensorRef;
use rayon::prelude::*;

use crate::inference::InferenceSession;
use crate::types::{
    BBox, DemographyResult, DetectedFace, Emotion, EmotionResult, FaceStateResult, Gender,
};
use crate::utils::image_utils::letterbox;

// --------------------------------------------------------------------
// 裁脸 helper — 统一实现
// --------------------------------------------------------------------

/// 按比例扩展 bbox 后裁脸并 clamp 到图像范围
///
/// 预处理约定:
///   - margin 为 bbox 宽高的小数比例, 四边各扩展 margin
///   - x1/y1 上限 0, x2/y2 下限图像宽高, 零面积裁剪报错
pub(crate) fn crop_face_with_margin(
    image: &DynamicImage,
    bbox: &BBox,
    margin: f32,
) -> Result<DynamicImage> {
    let (img_w, img_h) = image.dimensions();

    let w = bbox.width();
    let h = bbox.height();
    let dw = w * margin;
    let dh = h * margin;

    let x1 = (bbox.x1 - dw).max(0.0).round() as i32;
    let y1 = (bbox.y1 - dh).max(0.0).round() as i32;
    let x2 = (bbox.x2 + dw).min(img_w as f32).round() as i32;
    let y2 = (bbox.y2 + dh).min(img_h as f32).round() as i32;

    if x2 <= x1 || y2 <= y1 {
        anyhow::bail!(
            // bbox 与图像无重叠时报错
            "Bounding box ({},{})-({},{}) does not overlap the {}x{} image",
            bbox.x1,
            bbox.y1,
            bbox.x2,
            bbox.y2,
            img_w,
            img_h
        );
    }

    Ok(image.crop_imm(x1 as u32, y1 as u32, (x2 - x1) as u32, (y2 - y1) as u32))
}

/// 裁脸后强制 resize 到 (size x size), 不保持纵横比
///
/// 公共流程: 裁脸后强制缩放到目标尺寸
pub(crate) fn crop_and_resize(
    image: &DynamicImage,
    bbox: &BBox,
    margin: f32,
    size: u32,
) -> Result<DynamicImage> {
    let crop = crop_face_with_margin(image, bbox, margin)?;
    Ok(crop.resize_exact(size, size, image::imageops::FilterType::Lanczos3))
}

/// 通用 NCHW 张量构建: 逐像素填充并按通道归一化
pub(crate) fn crop_to_nchw(
    image: &DynamicImage,
    bbox: &BBox,
    margin: f32,
    size: u32,
    mean: Option<[f32; 3]>,
    std: Option<[f32; 3]>,
) -> Result<ndarray::Array4<f32>> {
    let resized = crop_and_resize(image, bbox, margin, size)?;
    let rgb = resized.to_rgb8();

    let mut arr = ndarray::Array4::<f32>::zeros((1, 3, size as usize, size as usize));
    for y in 0..size {
        for x in 0..size {
            let p = rgb.get_pixel(x, y);
            for c in 0..3 {
                let v = p[c] as f32 / 255.0;
                arr[[0, c, y as usize, x as usize]] = match (mean, std) {
                    (Some(m), Some(s)) => (v - m[c]) / s[c],
                    _ => v,
                };
            }
        }
    }
    Ok(arr)
}

/// 灰度 NCHW 裁剪 (1 通道): 用于 FER+ 这类单通道输入模型
pub(crate) fn crop_to_nchw_gray(
    image: &DynamicImage,
    bbox: &BBox,
    margin: f32,
    size: u32,
) -> Result<ndarray::Array4<f32>> {
    let resized = crop_and_resize(image, bbox, margin, size)?;
    let rgb = resized.to_rgb8();

    let mut arr = ndarray::Array4::<f32>::zeros((1, 1, size as usize, size as usize));
    for y in 0..size {
        for x in 0..size {
            let p = rgb.get_pixel(x, y);
            // ITU-R BT.601 亮度
            let gray = 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32;
            arr[[0, 0, y as usize, x as usize]] = gray / 255.0;
        }
    }
    Ok(arr)
}

// --------------------------------------------------------------------
// 纯函数后处理
// --------------------------------------------------------------------

/// softmax: exp(x - max) / sum, 数值稳定
pub(crate) fn softmax(values: &[f32]) -> Vec<f32> {
    if values.is_empty() {
        return Vec::new();
    }
    let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = values.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        return vec![1.0 / values.len() as f32; values.len()];
    }
    exps.iter().map(|e| e / sum).collect()
}

/// argmax 返回首个最大值索引
pub(crate) fn argmax(values: &[f32]) -> Option<usize> {
    let mut best: Option<usize> = None;
    for (i, &v) in values.iter().enumerate() {
        match best {
            None => best = Some(i),
            Some(b) => {
                if v > values[b] {
                    best = Some(i)
                }
            }
        }
    }
    best
}

/// AgeGender 后处理:
///   - prediction[:2] 为 gender logits → argmax (0=Female, 1=Male), 不做 softmax
///   - prediction[2] 为归一化年龄 → round(x * 100)
pub(crate) fn age_gender_postprocess(prediction: &[f32]) -> Result<(Gender, u32)> {
    if prediction.len() < 3 {
        anyhow::bail!(
            "AgeGender output too short: {} floats, expected >= 3",
            prediction.len()
        );
    }
    let gender = match argmax(&prediction[..2]) {
        Some(0) => Gender::Female,
        Some(1) => Gender::Male,
        _ => anyhow::bail!("invalid gender logits"),
    };
    // 四舍五入; 负值裁为 0
    let age = (prediction[2] * 100.0).round().max(0.0) as u32;
    Ok((gender, age))
}

/// AffectNet 情绪标签顺序
pub(crate) const EMOTION_LABELS: [Emotion; 8] = [
    Emotion::Neutral,
    Emotion::Happy,
    Emotion::Sad,
    Emotion::Surprise,
    Emotion::Fear,
    Emotion::Disgust,
    Emotion::Anger,
    Emotion::Contempt,
];

/// FER+ 情绪标签顺序 (微软 ONNX Model Zoo emotion-ferplus-8)
pub(crate) const FERPLUS_LABELS: [Emotion; 8] = [
    Emotion::Neutral,
    Emotion::Happy,
    Emotion::Surprise,
    Emotion::Sad,
    Emotion::Anger,
    Emotion::Disgust,
    Emotion::Fear,
    Emotion::Contempt,
];

/// 情绪模型变体
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmotionModel {
    /// AffectNet7 (TorchScript, 7类) — 原始实现默认
    AffectNet7,
    /// AffectNet8 (TorchScript, 8类) — 原始实现可选项
    AffectNet8,
    /// FER+ 8类 ONNX (emotion-ferplus-8, 64x64 灰度输入)
    /// — 本实现默认: AffectNet 系列仅发布 TorchScript, ORT 无法加载
    FerPlus,
}

impl EmotionModel {
    /// 类别数
    pub fn num_classes(self) -> usize {
        match self {
            EmotionModel::AffectNet7 => 7,
            _ => 8,
        }
    }

    /// 标签表
    pub fn labels(self) -> &'static [Emotion; 8] {
        match self {
            EmotionModel::FerPlus => &FERPLUS_LABELS,
            _ => &EMOTION_LABELS,
        }
    }
}

/// Emotion 后处理:
///   - softmax → argmax → 标签
///   - 7类 (AffectNet7) 输入: 只有前7个logit参与 softmax, 返回 None
pub(crate) fn emotion_postprocess(
    logits: &[f32],
    model: EmotionModel,
) -> Result<Option<EmotionResult>> {
    let num_classes = model.num_classes();
    let valid: &[f32] = match num_classes {
        0..=8 => &logits[..logits.len().min(num_classes)],
        _ => &logits[..8],
    };
    if valid.is_empty() {
        anyhow::bail!("Emotion logits are empty");
    }
    // AffectNet7: 模型无 Contempt 头, 若剩余 logit 数小于 7, 输出不完整
    if num_classes == 7 && valid.len() < 7 {
        return Ok(None);
    }
    let probs = softmax(valid);
    let idx = argmax(&probs).ok_or_else(|| anyhow::anyhow!("Emotion argmax failed"))?;
    Ok(Some(EmotionResult {
        emotion: model.labels()[idx],
        confidence: probs[idx],
    }))
}

/// FaceAttribNet 后处理:
/// 5 个独立概率, 非 softmax, clamp 到 [0, 1]
pub(crate) fn face_attrib_postprocess(prediction: &[f32]) -> Result<FaceStateResult> {
    if prediction.len() < 5 {
        anyhow::bail!(
            "FaceAttribNet output too short: {} floats, expected >= 5",
            prediction.len()
        );
    }
    let clamp01 = |v: f32| v.clamp(0.0, 1.0);
    Ok(FaceStateResult {
        left_eye_open: clamp01(prediction[0]),
        right_eye_open: clamp01(prediction[1]),
        eyeglasses: clamp01(prediction[2]),
        mask: clamp01(prediction[3]),
        sunglasses: clamp01(prediction[4]),
    })
}

// --------------------------------------------------------------------
// AgeGender
// --------------------------------------------------------------------

/// 年龄/性别预测器
///
/// - 输入: bbox 中心对齐裁脸 (1.5x 扩展) → 224x224 RGB [0,1]
/// - 输出: 3维向量 [gender_logit_f, gender_logit_m, normalized_age]
pub struct AgeGender {
    session: InferenceSession,
    input_size: u32,
}

impl AgeGender {
    /// 创建 AgeGender 预测器
    ///
    /// 输入尺寸从模型 input shape 动态读取 (genderage.onnx 为 96x96)
    pub fn from_session(session: InferenceSession) -> Result<Self> {
        let input_size = session
            .session()
            .inputs()
            .first()
            .and_then(|input| input.dtype().tensor_shape())
            .and_then(|shape| shape.get(3).copied())
            .filter(|&d| d > 0)
            .unwrap_or(224) as u32;
        info!("AgeGender: input_size={}x{}", input_size, input_size);
        Ok(Self {
            session,
            input_size,
        })
    }

    /// 预测年龄和性别, 结果写回 face
    pub fn predict(
        &mut self,
        image: &DynamicImage,
        face: &DetectedFace,
    ) -> Result<DemographyResult> {
        // 中心对齐裁剪: scale = 224 / (max(w,h)*1.5),
        // 即等价的 1.5x 边长方形 ROI (rotation=0)
        let (gender, age) = {
            let bbox = &face.bbox;
            let side = bbox.width().max(bbox.height()) * 1.5;
            let (cx, cy) = bbox.center();
            let roi = BBox::new(
                cx - side / 2.0,
                cy - side / 2.0,
                cx + side / 2.0,
                cy + side / 2.0,
            );
            let input = crop_to_nchw(image, &roi, 0.0, self.input_size, None, None)?;

            let input = TensorRef::from_array_view(&input)?;
            let output_name = self.session.session().outputs()[0].name().to_string();
            let outputs = self.session.session_mut().run(ort::inputs![input])?;
            let (_shape, data_ref) = outputs
                .get(output_name.as_str())
                .ok_or_else(|| {
                    anyhow::anyhow!("Attribute model output '{}' not found", output_name)
                })?
                .try_extract_tensor::<f32>()?;
            let data: Vec<f32> = data_ref.to_vec();
            drop(outputs);
            age_gender_postprocess(&data)?
        };

        // 注意: 由调用者负责把结果写回 face.gender / face.age;
        // Rust API 的 face 为不可变引用

        Ok(DemographyResult {
            gender,
            age: Some(age),
            age_group: None,
            race: None,
        })
    }

    /// 批量预测年龄和性别: 逐脸裁剪 + 一次 batch 推理
    pub fn predict_batch(
        &mut self,
        image: &DynamicImage,
        faces: &[&DetectedFace],
    ) -> Result<Vec<Option<DemographyResult>>> {
        if faces.is_empty() {
            return Ok(Vec::new());
        }
        let size = self.input_size;
        // 预处理互相独立, rayon 并行裁剪
        let prepped: Vec<Result<Array4<f32>>> = faces
            .par_iter()
            .map(|face| {
                let bbox = &face.bbox;
                let side = bbox.width().max(bbox.height()) * 1.5;
                let (cx, cy) = bbox.center();
                let roi = BBox::new(
                    cx - side / 2.0,
                    cy - side / 2.0,
                    cx + side / 2.0,
                    cy + side / 2.0,
                );
                crop_to_nchw(image, &roi, 0.0, size, None, None)
            })
            .collect();
        let mut inputs = Vec::with_capacity(faces.len());
        let mut valid = Vec::with_capacity(faces.len());
        for (i, r) in prepped.into_iter().enumerate() {
            match r {
                Ok(blob) => {
                    inputs.push(blob);
                    valid.push(i);
                }
                Err(e) => {
                    log::debug!("AgeGender: face {} preprocessing failed: {}", i, e);
                }
            }
        }

        let mut results: Vec<Option<DemographyResult>> = vec![None; faces.len()];
        if inputs.is_empty() {
            return Ok(results);
        }

        let n = inputs.len();
        let mut batch = ndarray::Array4::<f32>::zeros((n, 3, size as usize, size as usize));
        for (b, blob) in inputs.iter().enumerate() {
            batch
                .slice_mut(ndarray::s![b..b + 1, .., .., ..])
                .assign(blob);
        }

        let tensor = TensorRef::from_array_view(&batch)?;
        let output_name = self.session.session().outputs()[0].name().to_string();
        let outputs = self.session.session_mut().run(ort::inputs![tensor])?;
        let (_shape, data_ref) = outputs
            .get(output_name.as_str())
            .ok_or_else(|| anyhow::anyhow!("Attribute model output '{}' not found", output_name))?
            .try_extract_tensor::<f32>()?;
        let data: Vec<f32> = data_ref.to_vec();
        drop(outputs);

        let dim = data.len() / n;
        for (b, &i) in valid.iter().enumerate() {
            let row = &data[b * dim..(b + 1) * dim];
            if let Ok((gender, age)) = age_gender_postprocess(row) {
                results[i] = Some(DemographyResult {
                    gender,
                    age: Some(age),
                    age_group: None,
                    race: None,
                });
            }
        }
        Ok(results)
    }
}

// --------------------------------------------------------------------
// EmotionNet
// --------------------------------------------------------------------

/// 情绪识别预测器
///
/// - AffectNet 变体: 112x112, ImageNet 归一化 — 但该系列仅发布
///   TorchScript 权重, ORT 无法加载
/// - FER+ 变体 (本实现默认): emotion-ferplus-8.onnx, 64x64 灰度输入,
///   输出 8类 logits → softmax → argmax
pub struct EmotionNet {
    session: InferenceSession,
    input_size: u32,
    /// 模型变体 (决定标签顺序与预处理)
    model: EmotionModel,
    /// bbox 裁脸 margin (5 点对齐优先; 无对齐时退化为 margin 裁脸)
    margin: f32,
}

impl EmotionNet {
    /// 创建 Emotion 预测器
    pub fn from_session(session: InferenceSession, model: EmotionModel) -> Result<Self> {
        let input_size = match model {
            EmotionModel::FerPlus => 64,
            _ => 112,
        };
        info!(
            "EmotionNet: input_size={}x{}, model={:?}, num_classes={}",
            input_size,
            input_size,
            model,
            model.num_classes()
        );
        Ok(Self {
            session,
            input_size,
            model,
            margin: 0.2,
        })
    }

    /// 预测情绪, 结果写回 face
    pub fn predict(&mut self, image: &DynamicImage, face: &DetectedFace) -> Result<EmotionResult> {
        let result = {
            let input = match self.model {
                // FER+: 64x64 灰度 [0,1] (emotion-ferplus-8 输入为 1x1x64x64)
                EmotionModel::FerPlus => {
                    crop_to_nchw_gray(image, &face.bbox, self.margin, self.input_size)?
                }
                // AffectNet: 112x112, ImageNet mean/std 归一化
                // (参考实现用5点landmarks对齐; 此处用bbox裁脸近似)
                _ => crop_to_nchw(
                    image,
                    &face.bbox,
                    self.margin,
                    self.input_size,
                    Some([0.485, 0.456, 0.406]),
                    Some([0.229, 0.224, 0.225]),
                )?,
            };

            let input = TensorRef::from_array_view(&input)?;
            let output_name = self.session.session().outputs()[0].name().to_string();
            let outputs = self.session.session_mut().run(ort::inputs![input])?;
            let (_shape, data_ref) = outputs
                .get(output_name.as_str())
                .ok_or_else(|| {
                    anyhow::anyhow!("Attribute model output '{}' not found", output_name)
                })?
                .try_extract_tensor::<f32>()?;
            let data: Vec<f32> = data_ref.to_vec();
            drop(outputs);
            emotion_postprocess(&data, self.model)?.ok_or_else(|| {
                anyhow::anyhow!(
                    "Emotion output incomplete for {} classes",
                    self.model.num_classes()
                )
            })?
        };

        // 注意: 由调用者负责把结果写回 face.emotion / face.emotion_confidence;
        // Rust API 的 face 为不可变引用
        Ok(result)
    }

    /// 批量预测情绪: 逐脸预处理后拼 (N,C,H,W) 单次推理
    ///
    /// 返回按输入顺序对齐的结果; 预处理失败的脸为 None
    pub fn predict_batch(
        &mut self,
        image: &DynamicImage,
        faces: &[&DetectedFace],
    ) -> Result<Vec<Option<EmotionResult>>> {
        if faces.is_empty() {
            return Ok(Vec::new());
        }
        let size = self.input_size as usize;
        // 预处理互相独立, rayon 并行裁剪
        let prepped: Vec<Result<ndarray::Array4<f32>>> = faces
            .par_iter()
            .map(|face| match self.model {
                EmotionModel::FerPlus => {
                    crop_to_nchw_gray(image, &face.bbox, self.margin, self.input_size)
                }
                _ => crop_to_nchw(
                    image,
                    &face.bbox,
                    self.margin,
                    self.input_size,
                    Some([0.485, 0.456, 0.406]),
                    Some([0.229, 0.224, 0.225]),
                ),
            })
            .collect();
        let mut inputs = Vec::with_capacity(faces.len());
        let mut valid = Vec::with_capacity(faces.len());
        for (i, r) in prepped.into_iter().enumerate() {
            match r {
                Ok(blob) => {
                    inputs.push(blob);
                    valid.push(i);
                }
                Err(e) => {
                    log::debug!("EmotionNet: face {} preprocessing failed: {}", i, e);
                }
            }
        }
        let mut results: Vec<Option<EmotionResult>> = vec![None; faces.len()];
        if inputs.is_empty() {
            return Ok(results);
        }

        let n = inputs.len();
        let channels = inputs[0].dim().1;
        let mut batch = ndarray::Array4::<f32>::zeros((n, channels, size, size));
        for (b, blob) in inputs.iter().enumerate() {
            batch
                .slice_mut(ndarray::s![b..b + 1, .., .., ..])
                .assign(blob);
        }

        let tensor = TensorRef::from_array_view(&batch)?;
        let output_name = self.session.session().outputs()[0].name().to_string();
        let outputs = self.session.session_mut().run(ort::inputs![tensor])?;
        let (_shape, data_ref) = outputs
            .get(output_name.as_str())
            .ok_or_else(|| anyhow::anyhow!("Attribute model output '{}' not found", output_name))?
            .try_extract_tensor::<f32>()?;
        let data: Vec<f32> = data_ref.to_vec();
        drop(outputs);

        let dim = data.len() / n;
        for (b, &i) in valid.iter().enumerate() {
            let row = &data[b * dim..(b + 1) * dim];
            if let Ok(Some(er)) = emotion_postprocess(row, self.model) {
                results[i] = Some(er);
            }
        }
        Ok(results)
    }
}

// --------------------------------------------------------------------
// FaceAttribNet
// --------------------------------------------------------------------

/// 人脸状态预测器
///
/// - 输入: bbox (可选 margin 扩展) 裁脸 → letterbox 到 128x128, [0,1]
///   (mean/std 归一化已内嵌在 ONNX 图中)
/// - 输出: 5 个独立二分类概率 (左/右眼睁开、眼镜、口罩、墨镜)
pub struct FaceAttribNet {
    session: InferenceSession,
    input_size: u32,
    margin: f32,
}

impl FaceAttribNet {
    /// 创建 FaceAttribNet 预测器 (默认 input_size=(128,128), margin=0.0;
    /// 预处理用 letterbox + 居中零填充)
    pub fn from_session(session: InferenceSession, margin: f32) -> Result<Self> {
        let input_size = 128u32;
        info!(
            "FaceAttribNet: input_size={}x{}, margin={}",
            input_size, input_size, margin
        );
        Ok(Self {
            session,
            input_size,
            margin,
        })
    }

    /// 预测人脸状态, 结果写回 face
    pub fn predict(
        &mut self,
        image: &DynamicImage,
        face: &DetectedFace,
    ) -> Result<FaceStateResult> {
        let result = {
            // 裁脸 → letterbox (保持比例、居中零填充) → [0,1],
            // 无 mean/std (已内嵌模型)
            let crop = crop_face_with_margin(image, &face.bbox, self.margin)?;
            let (canvas, _scale, _pad_x, _pad_y) =
                letterbox(&crop, self.input_size, self.input_size);
            let size = self.input_size as usize;
            let mut input = ndarray::Array4::<f32>::zeros((1, 3, size, size));
            for y in 0..size {
                for x in 0..size {
                    let p = canvas.get_pixel(x as u32, y as u32);
                    for c in 0..3 {
                        input[[0, c, y, x]] = p[c] as f32 / 255.0;
                    }
                }
            }

            let input = TensorRef::from_array_view(&input)?;
            let output_name = self.session.session().outputs()[0].name().to_string();
            let outputs = self.session.session_mut().run(ort::inputs![input])?;
            let (_shape, data_ref) = outputs
                .get(output_name.as_str())
                .ok_or_else(|| {
                    anyhow::anyhow!("Attribute model output '{}' not found", output_name)
                })?
                .try_extract_tensor::<f32>()?;
            let data: Vec<f32> = data_ref.to_vec();
            drop(outputs);
            face_attrib_postprocess(&data)?
        };

        // 注意: 由调用者负责把结果写回 face.face_state 等五个字段;
        // Rust API 的 face 为不可变引用
        Ok(result)
    }

    /// 批量预测人脸状态: 逐脸 letterbox 裁剪 + 一次 batch 推理
    pub fn predict_batch(
        &mut self,
        image: &DynamicImage,
        faces: &[&DetectedFace],
    ) -> Result<Vec<Option<FaceStateResult>>> {
        if faces.is_empty() {
            return Ok(Vec::new());
        }
        let size = self.input_size as usize;
        // 预处理互相独立, rayon 并行裁剪
        let margin = self.margin;
        let input_size = self.input_size;
        let prepped: Vec<Result<ndarray::Array4<f32>>> = faces
            .par_iter()
            .map(|face| {
                let crop = crop_face_with_margin(image, &face.bbox, margin)?;
                let (canvas, _scale, _pad_x, _pad_y) = letterbox(&crop, input_size, input_size);
                let mut blob = ndarray::Array4::<f32>::zeros((1, 3, size, size));
                for y in 0..size {
                    for x in 0..size {
                        let p = canvas.get_pixel(x as u32, y as u32);
                        for c in 0..3 {
                            blob[[0, c, y, x]] = p[c] as f32 / 255.0;
                        }
                    }
                }
                Ok(blob)
            })
            .collect();
        let mut inputs = Vec::with_capacity(faces.len());
        let mut valid = Vec::with_capacity(faces.len());
        for (i, r) in prepped.into_iter().enumerate() {
            match r {
                Ok(blob) => {
                    inputs.push(blob);
                    valid.push(i);
                }
                Err(e) => {
                    log::debug!("FaceAttribNet: face {} preprocessing failed: {}", i, e);
                }
            }
        }

        let mut results: Vec<Option<FaceStateResult>> = vec![None; faces.len()];
        if inputs.is_empty() {
            return Ok(results);
        }

        let n = inputs.len();
        let mut batch = ndarray::Array4::<f32>::zeros((n, 3, size, size));
        for (b, blob) in inputs.iter().enumerate() {
            batch
                .slice_mut(ndarray::s![b..b + 1, .., .., ..])
                .assign(blob);
        }

        let tensor = TensorRef::from_array_view(&batch)?;
        let output_name = self.session.session().outputs()[0].name().to_string();
        let outputs = self.session.session_mut().run(ort::inputs![tensor])?;
        let (_shape, data_ref) = outputs
            .get(output_name.as_str())
            .ok_or_else(|| anyhow::anyhow!("Attribute model output '{}' not found", output_name))?
            .try_extract_tensor::<f32>()?;
        let data: Vec<f32> = data_ref.to_vec();
        drop(outputs);

        let dim = data.len() / n;
        for (b, &i) in valid.iter().enumerate() {
            let row = &data[b * dim..(b + 1) * dim];
            if let Ok(state) = face_attrib_postprocess(row) {
                results[i] = Some(state);
            }
        }
        Ok(results)
    }
}

// --------------------------------------------------------------------
// 单元测试
// --------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    /// 构建纯色测试图像
    fn test_image(w: u32, h: u32) -> DynamicImage {
        DynamicImage::ImageRgb8(RgbImage::from_pixel(w, h, image::Rgb([100u8, 150, 200])))
    }

    fn test_face() -> DetectedFace {
        DetectedFace {
            bbox: BBox::new(10.0, 10.0, 50.0, 50.0),
            confidence: 0.9,
            landmarks: None,
            mesh_3d: None,
            mesh_score: None,
            track_id: None,
            embedding: None,
            gender: None,
            age: None,
            age_group: None,
            race: None,
            emotion: None,
            emotion_confidence: None,
            face_state: None,
            quality: None,
            gaze: None,
            head_pose: None,
            spoofing: None,
        }
    }

    // ---- softmax ----

    #[test]
    fn test_softmax_orders_and_sums_to_one() {
        // softmax 应保序且和为 1
        let vals = [1.0, 3.0, 2.0];
        let probs = softmax(&vals);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum={}", sum);
        assert!(probs[1] > probs[2] && probs[2] > probs[0]);
        assert_eq!(argmax(&probs), Some(1));
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // 大 logit 不应溢出
        let vals = [1000.0, 1001.0];
        let probs = softmax(&vals);
        assert!(probs.iter().all(|p| p.is_finite()));
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_softmax_empty() {
        assert!(softmax(&[]).is_empty());
    }

    #[test]
    fn test_softmax_extreme_values() {
        // 全 -inf / NaN 输入: 退化为均匀分布或有限值, 不 panic
        let neg_inf = [f32::NEG_INFINITY; 4];
        let probs = softmax(&neg_inf);
        assert!(probs.iter().all(|p| (p - 0.25).abs() < 1e-6));
        let nans = [f32::NAN; 3];
        let probs = softmax(&nans);
        assert!(probs.iter().all(|p| p.is_finite()));
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    // ---- argmax ----

    #[test]
    fn test_argmax_first_max_wins() {
        // 并列最大取首个 (对齐 np.argmax)
        assert_eq!(argmax(&[1.0, 5.0, 5.0]), Some(1));
        assert_eq!(argmax(&[3.0, 3.0, 1.0]), Some(0));
        assert_eq!(argmax(&[]), None);
    }

    // ---- AgeGender 后处理 ----

    #[test]
    fn test_age_gender_postprocess_male() {
        // gender logits argmax → Male; age = round(0.35*100) = 35
        let (gender, age) = age_gender_postprocess(&[0.2, 2.7, 0.35]).unwrap();
        assert_eq!(gender, Gender::Male);
        assert_eq!(age, 35);
    }

    #[test]
    fn test_age_gender_postprocess_female() {
        let (gender, age) = age_gender_postprocess(&[3.1, 0.2, 0.426]).unwrap();
        assert_eq!(gender, Gender::Female);
        assert_eq!(age, 43); // round(42.6)
    }

    #[test]
    fn test_age_gender_postprocess_negative_age_clamped() {
        // 负归一化年龄 clamp 到 0 (u32 语义)
        let (gender, age) = age_gender_postprocess(&[1.0, -1.0, -0.05]).unwrap();
        assert_eq!(gender, Gender::Female);
        assert_eq!(age, 0);
    }

    #[test]
    fn test_age_gender_postprocess_short_output() {
        assert!(age_gender_postprocess(&[1.0, 0.0]).is_err());
    }

    // ---- Emotion 后处理 ----

    #[test]
    fn test_emotion_postprocess_8class_order() {
        // 8类: 每个独大 logit 应映射到标签顺序的对应类别
        for (i, expected) in EMOTION_LABELS.iter().enumerate() {
            let mut logits = vec![0.0f32; 8];
            logits[i] = 10.0;
            let result = emotion_postprocess(&logits, EmotionModel::AffectNet8)
                .unwrap()
                .unwrap();
            assert_eq!(result.emotion, *expected, "index {} mislabeled", i);
            assert!(result.confidence > 0.99);
        }
    }

    #[test]
    fn test_emotion_postprocess_ferplus_order() {
        // FER+ 标签顺序与 AffectNet 不同 (Happy/Surprise/Sad/Anger/Disgust/Fear 位次互换)
        for (i, expected) in FERPLUS_LABELS.iter().enumerate() {
            let mut logits = vec![0.0f32; 8];
            logits[i] = 10.0;
            let result = emotion_postprocess(&logits, EmotionModel::FerPlus)
                .unwrap()
                .unwrap();
            assert_eq!(result.emotion, *expected, "index {} mislabeled", i);
        }
    }

    #[test]
    fn test_emotion_postprocess_7class_ignores_8th_logit() {
        // AffectNet7: 第 8 个 logit (Contempt) 不参与 softmax
        let logits = [0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0];
        let result = emotion_postprocess(&logits, EmotionModel::AffectNet7)
            .unwrap()
            .unwrap();
        assert_eq!(result.emotion, Emotion::Happy);
    }

    #[test]
    fn test_emotion_postprocess_7class_confidence() {
        let logits = [1.0, 4.0, 2.0, 0.5, 0.5, 0.5, 0.5];
        let result = emotion_postprocess(&logits, EmotionModel::AffectNet7)
            .unwrap()
            .unwrap();
        assert_eq!(result.emotion, Emotion::Happy);
        assert!(result.confidence > 0.5 && result.confidence <= 1.0);
        // 手工 softmax 对照 / hand-computed softmax cross-check
        let max = 4.0f32;
        let exps: Vec<f32> = logits.iter().map(|v| (v - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let want = exps[1] / sum;
        assert!((result.confidence - want).abs() < 1e-6);
    }

    #[test]
    fn test_emotion_postprocess_incomplete_7class() {
        // 7类但输出不足 7 个 logit → None (模型输出不完整)
        assert!(
            emotion_postprocess(&[0.1, 0.2, 0.3], EmotionModel::AffectNet7)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn test_emotion_postprocess_empty() {
        assert!(emotion_postprocess(&[], EmotionModel::AffectNet8).is_err());
    }

    // ---- FaceAttribNet 后处理 ----

    #[test]
    fn test_face_attrib_postprocess_clamps() {
        // 概率 clamp 到 [0, 1]; 输出顺序固定为
        // 左眼/右眼/眼镜/口罩/墨镜
        let r = face_attrib_postprocess(&[-0.5, 1.2, 0.9, 0.3, 2.0]).unwrap();
        assert_eq!(r.left_eye_open, 0.0);
        assert_eq!(r.right_eye_open, 1.0);
        assert_eq!(r.eyeglasses, 0.9);
        assert_eq!(r.mask, 0.3);
        assert_eq!(r.sunglasses, 1.0);
    }

    #[test]
    fn test_face_attrib_postprocess_exact_values() {
        let r = face_attrib_postprocess(&[0.1, 0.2, 0.3, 0.4, 0.5]).unwrap();
        assert_eq!(r.left_eye_open, 0.1);
        assert_eq!(r.right_eye_open, 0.2);
        assert_eq!(r.eyeglasses, 0.3);
        assert_eq!(r.mask, 0.4);
        assert_eq!(r.sunglasses, 0.5);
    }

    #[test]
    fn test_face_attrib_postprocess_short_output() {
        assert!(face_attrib_postprocess(&[0.1, 0.2, 0.3]).is_err());
    }

    // ---- 裁脸 helper ----

    #[test]
    fn test_crop_face_with_margin_clamps_to_image() {
        // margin 扩展超出图像边界应被 clamp, 不报错
        let img = test_image(40, 40);
        let face = test_face();
        let crop = crop_face_with_margin(&img, &face.bbox, 0.5).unwrap();
        // bbox (10,10)-(50,50) margin 0.5 → (-10,-10)-(70,70) → clamp 到 (0,0)-(40,40)
        assert_eq!(crop.dimensions(), (40, 40));
    }

    #[test]
    fn test_crop_face_with_margin_no_overlap_errors() {
        // bbox 完全在图像外应报错
        let img = test_image(40, 40);
        let bbox = BBox::new(60.0, 60.0, 80.0, 80.0);
        assert!(crop_face_with_margin(&img, &bbox, 0.0).is_err());
    }

    #[test]
    fn test_crop_and_resize_output_size() {
        let img = test_image(100, 80);
        let face = test_face();
        let out = crop_and_resize(&img, &face.bbox, 0.0, 224).unwrap();
        assert_eq!(out.dimensions(), (224, 224));
    }

    #[test]
    fn test_crop_to_nchw_shape_and_normalization() {
        // 无归一化: 值 ∈ [0,1]; 有归一化: 中心像素按 (v-mean)/std
        let img = test_image(80, 80);
        let face = test_face();
        let arr = crop_to_nchw(&img, &face.bbox, 0.0, 32, None, None).unwrap();
        assert_eq!(arr.shape(), &[1, 3, 32, 32]);
        assert!((arr[[0, 0, 16, 16]] - 100.0 / 255.0).abs() < 1e-6);

        let arr = crop_to_nchw(
            &img,
            &face.bbox,
            0.0,
            32,
            Some([0.485, 0.456, 0.406]),
            Some([0.229, 0.224, 0.225]),
        )
        .unwrap();
        let v = 100.0 / 255.0;
        assert!((arr[[0, 0, 16, 16]] - (v - 0.485) / 0.229).abs() < 1e-6);
        assert!((arr[[0, 2, 16, 16]] - (200.0 / 255.0 - 0.406) / 0.225).abs() < 1e-6);
    }
}
