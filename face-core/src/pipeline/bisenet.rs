// ====================================================================
// BiSeNet 面部语义分割 — 19类面部区域解析
// ====================================================================
//
// 模型: BiSeNet ResNet18 (CelebAMask-HQ 数据集训练)
//       - 输入: 512x512 RGB, ImageNet 归一化
//       - 输出: (1, 19, 512, 512) 每个像素的19类logits
//       - 19类标签: 背景/皮肤/眉毛/眼睛/鼻子/嘴唇/头发等
//       - 用途: 美颜时精确定位 皮肤区域(磨皮) / 嘴唇(口红) / 眼睛(大眼) 等

use anyhow::Result;
use image::{DynamicImage, GenericImageView};
use log::{debug, info, trace};
use ort::value::TensorRef;

use crate::inference::InferenceSession;
use crate::types::parsing_labels;
use crate::types::{DetectedFace, ParsingMask};
use crate::utils::image_utils::resize_mask;

/// ImageNet 均值 (RGB)
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
/// ImageNet 标准差 (RGB)
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// BiSeNet 面部语义分割器
pub struct BiSeNet {
    session: InferenceSession,
    input_size: (u32, u32),
}

impl BiSeNet {
    /// 创建 BiSeNet 解析器
    pub fn from_session(session: InferenceSession) -> Result<Self> {
        let input_size = (512u32, 512u32);
        info!(
            "BiSeNet: input_size={}x{} (19 classes)",
            input_size.0, input_size.1
        );
        Ok(Self {
            session,
            input_size,
        })
    }

    /// 对单个人脸执行语义分割
    pub fn parse(&mut self, image: &DynamicImage, face: &DetectedFace) -> Result<ParsingMask> {
        let expanded_bbox = face.bbox.expand((0.2, 0.4, 0.2, 0.2));

        let (img_w, img_h) = image.dimensions();
        let clamped = expanded_bbox.clamp(img_w as f32, img_h as f32);
        let (x1, y1, x2, y2) = clamped.to_int();
        let crop_w = (x2 - x1) as u32;
        let crop_h = (y2 - y1) as u32;

        if crop_w == 0 || crop_h == 0 {
            anyhow::bail!("Invalid crop region for face parsing");
        }

        let face_crop = image.crop_imm(x1 as u32, y1 as u32, crop_w, crop_h);

        let input_array = self.preprocess_crop(&face_crop)?;

        let input = TensorRef::from_array_view(&input_array)?;
        let output_name = self.session.session().outputs()[0].name().to_string();
        let outputs = self.session.session_mut().run(ort::inputs![input])?;

        // 提取输出数据到 owned Vec (避免借用冲突)
        let (_shape, logits_ref) = outputs
            .get(output_name.as_str())
            .ok_or_else(|| anyhow::anyhow!("BiSeNet model output '{}' not found", output_name))?
            .try_extract_tensor::<f32>()?;
        let logits_data: Vec<f32> = logits_ref.to_vec();
        drop(outputs);

        let mask = Self::postprocess(&logits_data, crop_w, crop_h);

        trace!(
            "BiSeNet: parsed face crop {}x{} -> parsing mask {}x{}",
            crop_w, crop_h, crop_w, crop_h
        );

        Ok(mask)
    }

    /// 批量处理多个人脸
    pub fn parse_batch(
        &mut self,
        image: &DynamicImage,
        faces: &[DetectedFace],
    ) -> Result<Vec<ParsingMask>> {
        let mut results = Vec::with_capacity(faces.len());
        for face in faces {
            match self.parse(image, face) {
                Ok(mask) => results.push(mask),
                Err(e) => {
                    debug!("BiSeNet: failed to parse face bbox={:?}: {}", face.bbox, e);
                }
            }
        }
        info!(
            "BiSeNet: batch parsed {}/{} faces",
            results.len(),
            faces.len()
        );
        Ok(results)
    }

    /// 裁剪区域的预处理
    fn preprocess_crop(&self, face_crop: &DynamicImage) -> Result<ndarray::Array4<f32>> {
        let (w, h) = self.input_size;
        let mut arr = ndarray::Array4::<f32>::zeros((1, 3, h as usize, w as usize));

        let resized = face_crop.resize_exact(w, h, image::imageops::FilterType::Lanczos3);
        let rgb = resized.to_rgb8();

        for y in 0..h {
            for x in 0..w {
                let pixel = rgb.get_pixel(x, y);
                arr[[0, 0, y as usize, x as usize]] = pixel[0] as f32 / 255.0;
                arr[[0, 1, y as usize, x as usize]] = pixel[1] as f32 / 255.0;
                arr[[0, 2, y as usize, x as usize]] = pixel[2] as f32 / 255.0;
            }
        }

        // ImageNet 归一化
        for c in 0..3 {
            let m = IMAGENET_MEAN[c];
            let s = IMAGENET_STD[c];
            for y in 0..h {
                for x in 0..w {
                    arr[[0, c, y as usize, x as usize]] =
                        (arr[[0, c, y as usize, x as usize]] - m) / s;
                }
            }
        }

        trace!(
            "BiSeNet: preprocessed face crop -> (1,3,{},{}), ImageNet norm",
            w, h
        );

        Ok(arr)
    }

    /// 后处理: argmax 得到分类标签, 然后 resize 回原始尺寸
    fn postprocess(logits: &[f32], out_w: u32, out_h: u32) -> ParsingMask {
        let mask_h = 512usize;
        let mask_w = 512usize;

        let mut raw_mask = vec![0u8; mask_w * mask_h];
        for y in 0..mask_h {
            for x in 0..mask_w {
                let mut max_class: u8 = 0;
                let mut max_val = f32::NEG_INFINITY;
                for c in 0..19 {
                    let val = logits[(c * mask_h + y) * mask_w + x];
                    if val > max_val {
                        max_val = val;
                        max_class = c as u8;
                    }
                }
                raw_mask[y * mask_w + x] = max_class;
            }
        }

        let resized_mask = resize_mask(&raw_mask, mask_w as u32, mask_h as u32, out_w, out_h);

        let mask_arr =
            ndarray::Array2::from_shape_vec((out_h as usize, out_w as usize), resized_mask)
                .expect("Failed to create parsing mask array");

        ParsingMask {
            mask: mask_arr,
            width: out_w,
            height: out_h,
        }
    }
}

impl ParsingMask {
    /// 获取指定区域类型的二值遮罩
    pub fn get_region_mask(&self, region: &crate::types::FaceRegion) -> Vec<u8> {
        let label = region.label_id();
        let flat = self.mask.as_slice().unwrap();
        let n = self.width as usize * self.height as usize;
        let mut binary = vec![0u8; n];
        for i in 0..n {
            if flat[i] == label {
                binary[i] = 1;
            }
        }
        binary
    }

    /// 获取皮肤遮罩 (磨皮/美白的关键区域)
    pub fn get_skin_mask(&self) -> Vec<u8> {
        self.get_region_mask(&crate::types::FaceRegion::Skin)
    }

    /// 获取嘴唇遮罩 (口红/唇彩的关键区域)
    pub fn get_lips_mask(&self) -> Vec<u8> {
        let flat = self.mask.as_slice().unwrap();
        let n = self.width as usize * self.height as usize;
        let mut binary = vec![0u8; n];
        for i in 0..n {
            let label = flat[i];
            if label == parsing_labels::UPPER_LIP || label == parsing_labels::LOWER_LIP {
                binary[i] = 1;
            }
        }
        binary
    }

    /// 获取眼部遮罩 (大眼/眼影的关键区域)
    pub fn get_eyes_mask(&self) -> Vec<u8> {
        let flat = self.mask.as_slice().unwrap();
        let n = self.width as usize * self.height as usize;
        let mut binary = vec![0u8; n];
        for i in 0..n {
            let label = flat[i];
            if label == parsing_labels::LEFT_EYE || label == parsing_labels::RIGHT_EYE {
                binary[i] = 1;
            }
        }
        binary
    }
}
