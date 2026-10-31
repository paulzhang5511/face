// ====================================================================
// 管线模块入口 — 汇集全部人脸分析子管线
//
//   detector        人脸检测 (两种检测器)
//   detector_scrfd  SCRFD 检测器实现
//   facemesh        稠密三维人脸网格
//   bisenet         人脸语义分割
//   recognition     人脸识别特征提取
//   attribute       人脸属性 (性别/年龄/情绪/状态)
//   gaze            视线估计
//   headpose        头部姿态估计
//   quality         人脸图像质量评估
//   spoofing        活体检测
//   matting         人像抠图
//   tracking        多目标追踪
//   privacy         人脸匿名化
//   stores          人脸特征向量库
//   render          结果可视化绘制
// ====================================================================

pub mod bisenet;
pub mod detector;
pub mod detector_scrfd;
pub mod facemesh;
pub mod privacy;
pub mod quality;
pub mod recognition;
pub mod render;
pub mod spoofing;
pub mod stores;
pub mod tracking;

pub mod attribute;
pub mod gaze;
pub mod headpose;
pub mod matting;
