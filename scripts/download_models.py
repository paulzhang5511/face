#!/usr/bin/env python3
"""
模型下载脚本 — 下载美颜引擎需要的所有 ONNX 模型
Model Download Script — download all ONNX models required by the beauty engine

将下载的模型放到 models/ 目录, 编译时通过 include_bytes! 内嵌到 Rust 二进制中。
Place downloaded models in models/ directory; embedded into Rust binary via include_bytes! at compile time.

需要 Python 3.8+ 环境和 requests 库。
Requires Python 3.8+ and the requests library.

用法 / Usage:
    python scripts/download_models.py
"""

import hashlib
import os
import sys

try:
    import requests
except ImportError:
    print("Error: 'requests' library required. Install with: pip install requests")
    sys.exit(1)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# Hugging Face 权重仓库 (对齐 uniface/constants.py HF_MIRROR_URL) / HF weights repo
HF_REPO = "https://huggingface.co/yakhyo/uniface-weights/resolve/main"
# HF 国内镜像 / HF mirror for mainland China
HF_MIRROR = os.environ.get("BEAUTY_HF_MIRROR", "https://hf-mirror.com")

ONNX_MIME = "application/octet-stream"

# 模型下载列表 / Model download list
# url: 主源 (HF 仓库); save_as: 本地文件名 (须与 src/models/mod.rs / build.rs 一致)
MODELS = [
    # BlazeFace 短距离人脸检测器 / BlazeFace short-range face detector
    {
        "url": f"{HF_REPO}/blazeface.onnx",
        "save_as": "blazeface_128.onnx",
        "description": "BlazeFace face detector (460KB, 128x128)",
    },
    # FaceMesh V1 468点3D网格 / FaceMesh V1 468-point 3D mesh
    {
        "url": f"{HF_REPO}/face_mesh.onnx",
        "save_as": "facemesh_468.onnx",
        "description": "MediaPipe FaceMesh V1 468 landmarks (192x192)",
    },
    # BiSeNet ResNet18 面部语义分割 / BiSeNet ResNet18 face parsing
    {
        "url": f"{HF_REPO}/parsing_resnet18.onnx",
        "save_as": "bisenet_r18.onnx",
        "description": "BiSeNet ResNet18 face parsing 19-class (512x512)",
    },
    # SCRFD 500M 人脸检测 (5点关键点) / SCRFD 500M face detector (5-pt landmarks)
    # 注: HF 仓库无 kps 变体, 用标准 scrfd_500m (输出含 kps 分支, use_kps=true) / HF repo ships standard scrfd_500m whose outputs include the kps branch
    {
        "url": f"{HF_REPO}/scrfd_500m.onnx",
        "save_as": "scrfd_500m_kps.onnx",
        "description": "SCRFD 500M face detector with 5-point landmarks (640x640)",
    },
    # ArcFace MobileNetFace 人脸识别 / ArcFace MobileNetFace recognition
    {
        "url": f"{HF_REPO}/arcface_mnet.onnx",
        "save_as": "arcface_mbf.onnx",
        "description": "ArcFace MobileNetFace w600k recognition model (112x112)",
    },
    # AgeGender 年龄/性别预测 / AgeGender age/gender prediction
    {
        "url": f"{HF_REPO}/age_gender.onnx",
        "save_as": "genderage.onnx",
        "description": "AgeGender age/gender prediction model (224x224)",
    },
    # FER+ 情绪识别 / FER+ emotion recognition
    {
        "url": "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx",
        "save_as": "emotion-ferplus-8.onnx",
        "description": "FER+ emotion recognition model (64x64 grayscale, 8 classes)",
    },
    # FaceAttribNet 人脸状态 / FaceAttribNet face states
    {
        "url": f"{HF_REPO}/face_attrib_net.onnx",
        "save_as": "face_attrib_net.onnx",
        "description": "FaceAttribNet face-state model (128x128, 5 attributes)",
    },
    # MobileNetV2 视线估计 / MobileNetV2 gaze estimation
    {
        "url": f"{HF_REPO}/gaze_mobilenetv2.onnx",
        "save_as": "mobilenetv2_gaze.onnx",
        "description": "MobileNetV2 gaze estimation model (224x224)",
    },
    # ResNet18 头部姿态估计 / ResNet18 head pose estimation
    {
        "url": f"{HF_REPO}/headpose_resnet18.onnx",
        "save_as": "resnet18_headpose.onnx",
        "description": "ResNet18 head pose estimation model (224x224)",
    },
    # eDifFIQA 人脸质量评估 / eDifFIQA quality assessment
    {
        "url": f"{HF_REPO}/ediffiqa_t.onnx",
        "save_as": "ediffiqa.onnx",
        "description": "eDifFIQA face image quality assessment (112x112, 5-pt aligned)",
    },
    # MiniFASNet 活体检测 / MiniFASNet anti-spoofing
    {
        "url": f"{HF_REPO}/minifasnet_v2.onnx",
        "save_as": "minifasnet.onnx",
        "description": "MiniFASNet anti-spoofing model (80x80)",
    },
    # MODNet 人像抠图 / MODNet portrait matting
    {
        "url": f"{HF_REPO}/modnet_photographic.onnx",
        "save_as": "modnet.onnx",
        "description": "MODNet portrait matting model (shorter side 512)",
    },
]


def compute_sha256(filepath: str) -> str:
    """计算文件的 SHA-256 哈希值 / Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download_model(model_info: dict) -> bool:
    """下载单个模型 / Download a single model.

    主源失败时自动回退到 HF 镜像 (hf-mirror.com)。
    Falls back to the HF mirror (hf-mirror.com) when the primary URL fails.
    """
    save_path = os.path.join(MODELS_DIR, model_info["save_as"])

    if os.path.exists(save_path):
        expected = model_info.get("sha256")
        if expected is None or compute_sha256(save_path) == expected:
            print(f"  [SKIP] {model_info['save_as']} — already downloaded")
            return True
        print(f"  [WARN] {model_info['save_as']} exists but SHA256 mismatch, re-downloading...")

    # 候选 URL: 主源 + 镜像 (HF 路径替换为镜像域名)
    # Candidate URLs: primary + mirror (HF host swapped for mirror domain)
    basename = model_info["url"].rsplit("/", 1)[-1]
    candidates = [model_info["url"], f"{HF_MIRROR}/yakhyo/uniface-weights/resolve/main/{basename}"]

    # 企业代理做 TLS 拦截时, 证书链校验会失败; BEAUTY_SKIP_TLS_VERIFY=1 跳过校验 (仅下载用)
    # Corporate proxies doing TLS interception break cert-chain verification;
    # set BEAUTY_SKIP_TLS_VERIFY=1 to skip (download only)
    verify_tls = os.environ.get("BEAUTY_SKIP_TLS_VERIFY", "0") != "1"

    print(f"  [DOWNLOAD] {model_info['save_as']} — {model_info['description']}")

    for url in candidates:
        try:
            resp = requests.get(url, timeout=120, stream=True, verify=verify_tls)
            resp.raise_for_status()

            total_size = int(resp.headers.get("content-length", 0))
            downloaded = 0

            with open(save_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = downloaded * 100 / total_size
                        print(f"\r    {downloaded}/{total_size} bytes ({percent:.1f}%)", end="")

            print()  # newline

            # 验证 SHA-256 (仅在条目提供时) / Verify SHA-256 (only when provided)
            expected = model_info.get("sha256")
            if expected is not None:
                actual_sha = compute_sha256(save_path)
                if actual_sha != expected:
                    print(f"    [ERROR] SHA256 mismatch! Expected {expected}, got {actual_sha}")
                    os.remove(save_path)
                    return False

            file_size = os.path.getsize(save_path)
            print(f"    [OK] {model_info['save_as']} ({file_size/1024:.1f}KB)")
            return True

        except requests.RequestException as e:
            print(f"    [RETRY] via {url.split('/')[2]} failed: {e}")
            if os.path.exists(save_path):
                os.remove(save_path)
            continue

    return False


def main():
    """主入口 / Main entry point."""
    print("=" * 60)
    print("  beauty-rs Model Download Script")
    print("  Download ONNX models for embedded face beauty engine")
    print("=" * 60)

    os.makedirs(MODELS_DIR, exist_ok=True)

    success_count = 0
    fail_count = 0

    for model in MODELS:
        if download_model(model):
            success_count += 1
        else:
            fail_count += 1

    print(f"\nSummary: {success_count} succeeded, {fail_count} failed")

    if fail_count > 0:
        print("\nSome models failed to download. Please check your network connection")
        print("and retry. You can also manually download them and place in the models/ directory.")
        sys.exit(1)
    else:
        print("\nAll models downloaded successfully!")
        print(f"Models directory: {os.path.abspath(MODELS_DIR)}")
        print("You can now run: cargo build --release")


if __name__ == "__main__":
    main()