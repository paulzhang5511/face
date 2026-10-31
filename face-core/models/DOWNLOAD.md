# 模型手动下载清单 / Manual Model Download List

所有模型来自 Hugging Face 仓库 `yakhyo/uniface-weights`（UniFace 官方权重）。
下载后放入本仓库的 `models/` 目录，文件名必须与下表"保存名"一致。

仓库地址 / Repo URL:
- 主源: https://huggingface.co/yakhyo/uniface-weights/tree/main
- 国内镜像: https://hf-mirror.com/yakhyo/uniface-weights/tree/main

> 注: 若所在网络无法直连 Hugging Face，可自行配置 HTTP/HTTPS 代理后下载；本仓库不内置任何代理地址。

## 下载清单 / Download List (13 files)

| # | 保存名 (models/ 下) | 源文件名 | 下载 URL | 大小 | 用途 |
|---|---|---|---|---|---|
| 1 | blazeface_128.onnx | blazeface.onnx | https://hf-mirror.com/yakhyo/uniface-weights/resolve/main/blazeface.onnx | 470KB | BlazeFace 人脸检测 (128x128) |
| 2 | facemesh_468.onnx | face_mesh.onnx | https://hf-mirror.com/yakhyo/uniface-weights/resolve/main/face_mesh.onnx | 2.5MB | FaceMesh 468点网格 (192x192) |
| 3 | bisenet_r18.onnx | parsing_resnet18.onnx | https://hf-mirror.com/yakhyo/uniface-weights/resolve/main/parsing_resnet18.onnx | 53MB | BiSeNet 面部19类分割 (512x512) |
| 4 | scrfd_500m_kps.onnx | scrfd_500m.onnx | https://hf-mirror.com/yakhyo/uniface-weights/resolve/main/scrfd_500m.onnx | 2.5MB | SCRFD 检测器, 5点关键点 (640x640) |
| 5 | arcface_mbf.onnx | arcface_mnet.onnx | https://hf-mirror.com/yakhyo/uniface-weights/resolve/main/arcface_mnet.onnx | 13.6MB | ArcFace 人脸识别 (112x112) |
| 6 | genderage.onnx | age_gender.onnx | https://hf-mirror.com/yakhyo/uniface-weights/resolve/main/age_gender.onnx | 1.3MB | 年龄/性别预测 (224x224) |
| 7 | affectnet8.onnx | affecnet8.script | https://hf-mirror.com/yakhyo/uniface-weights/resolve/main/affecnet8.script | 18.5MB | AffectNet8 情绪识别 (112x112, 8类) |
| 8 | face_attrib_net.onnx | face_attrib_net.onnx | https://hf-mirror.com/yakhyo/uniface-weights/resolve/main/face_attrib_net.onnx | 43MB | FaceAttribNet 人脸状态 (128x128, 5属性) |
| 9 | mobilenetv2_gaze.onnx | gaze_mobilenetv2.onnx | https://hf-mirror.com/yakhyo/uniface-weights/resolve/main/gaze_mobilenetv2.onnx | 9.8MB | MobileNetV2 视线估计 (224x224) |
| 10 | resnet18_headpose.onnx | headpose_resnet18.onnx | https://hf-mirror.com/yakhyo/uniface-weights/resolve/main/headpose_resnet18.onnx | 44.7MB | ResNet18 头部姿态 (224x224) |
| 11 | ediffiqa.onnx | ediffiqa_t.onnx | https://hf-mirror.com/yakhyo/uniface-weights/resolve/main/ediffiqa_t.onnx | 6.9MB | eDifFIQA 人脸质量评估 (112x112) |
| 12 | minifasnet.onnx | minifasnet_v2.onnx | https://hf-mirror.com/yakhyo/uniface-weights/resolve/main/minifasnet_v2.onnx | 1.7MB | MiniFASNet 活体检测 (80x80) |
| 13 | modnet.onnx | modnet_photographic.onnx | https://hf-mirror.com/yakhyo/uniface-weights/resolve/main/modnet_photographic.onnx | 26MB | MODNet 人像抠图 (短边512) |

**合计 / Total: ~223MB**

## 一键下载脚本 / One-shot Download

PowerShell:

```powershell
$models = @(
    @{save="blazeface_128.onnx";    src="blazeface.onnx"},
    @{save="facemesh_468.onnx";     src="face_mesh.onnx"},
    @{save="bisenet_r18.onnx";      src="parsing_resnet18.onnx"},
    @{save="scrfd_500m_kps.onnx";   src="scrfd_500m.onnx"},
    @{save="arcface_mbf.onnx";      src="arcface_mnet.onnx"},
    @{save="genderage.onnx";        src="age_gender.onnx"},
    @{save="affectnet8.onnx";       src="affecnet8.script"},
    @{save="face_attrib_net.onnx";  src="face_attrib_net.onnx"},
    @{save="mobilenetv2_gaze.onnx"; src="gaze_mobilenetv2.onnx"},
    @{save="resnet18_headpose.onnx";src="headpose_resnet18.onnx"},
    @{save="ediffiqa.onnx";         src="ediffiqa_t.onnx"},
    @{save="minifasnet.onnx";       src="minifasnet_v2.onnx"},
    @{save="modnet.onnx";           src="modnet_photographic.onnx"}
)
foreach ($m in $models) {
    $url = "https://hf-mirror.com/yakhyo/uniface-weights/resolve/main/$($m.src)"
    $out = "models/$($m.save)"
    curl.exe --ssl-no-revoke -L --retry 3 -o $out $url
    if ($LASTEXITCODE -eq 0) { Write-Host "OK: $out" -ForegroundColor Green }
    else { Write-Host "FAIL: $url" -ForegroundColor Red }
}
```

## 验证 / Verify

下载完成后, 用以下命令确认（占位文件只有 104 字节, 真实模型均 > 400KB）:

```powershell
Get-ChildItem models\*.onnx | Where-Object Length -lt 1024   # 应输出为空
```

然后重新构建并跑端到端:

```powershell
cargo build --release
cargo test
.\target\release\face-demo <一张人脸照片路径>
```

## 注意 / Notes

- HF 仓库名为 `affecnet8.script`（原作者拼写如此）, 保存时须重命名为 `affectnet8.onnx`
- `scrfd_500m.onnx` 为标准版（输出含 kps 分支）, 保存为 `scrfd_500m_kps.onnx`
- SHA256 校验值未提供（HF 仓库未发布 checksums）, 用文件大小核对即可
