#!/usr/bin/env bash
# =====================================================================
# Android 构建脚本 — 使用 cargo-ndk 交叉编译到 Android 目标
# Android Build Script — cross-compile to Android targets using cargo-ndk
# =====================================================================
#
# 前置条件 / Prerequisites:
#   1. 安装 Rust Android 目标: rustup target add aarch64-linux-android armv7-linux-androideabi x86_64-linux-android i686-linux-android
#   2. 安装 cargo-ndk: cargo install cargo-ndk
#   3. 设置 ANDROID_NDK_HOME 环境变量
#   4. 安装 Python 3.8+ 用于模型下载脚本
#
# Usage:
#   chmod +x scripts/build_android.sh
#   ./scripts/build_android.sh [release|debug]

set -euo pipefail

BUILD_TYPE="${1:-release}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/models"
OUTPUT_DIR="$PROJECT_DIR/target/android"

# Android NDK 目标 / Android NDK targets
TARGETS=(
    "aarch64-linux-android"       # ARM64 (主流 / mainstream)
    "armv7-linux-androideabi"      # ARM32 (旧设备 / old devices)
    "x86_64-linux-android"        # x86_64 (模拟器 / emulator)
)

echo "===================================================================="
echo "  face Android Build Script"
echo "  Build type: $BUILD_TYPE"
echo "===================================================================="

# 检查模型文件 / Check model files
echo ""
echo "[1/4] Checking model files..."
REQUIRED_MODELS=("blazeface_128.onnx" "facemesh_468.onnx" "bisenet_r18.onnx")
MISSING=0
for model in "${REQUIRED_MODELS[@]}"; do
    if [ ! -f "$MODELS_DIR/$model" ]; then
        echo "  [MISSING] $model — run 'python scripts/download_models.py' first"
        MISSING=1
    else
        echo "  [OK] $model"
    fi
done
if [ $MISSING -eq 1 ]; then
    echo ""
    echo "Some models are missing. Download them with:"
    echo "  python scripts/download_models.py"
    exit 1
fi

# 检查 NDK / Check NDK
echo ""
echo "[2/4] Checking Android NDK..."
if [ -z "${ANDROID_NDK_HOME:-}" ]; then
    echo "  [ERROR] ANDROID_NDK_HOME is not set"
    echo "  Please set it to your Android NDK path, e.g.:"
    echo "    export ANDROID_NDK_HOME=~/Android/Sdk/ndk/27.0.12077973"
    exit 1
fi
echo "  [OK] ANDROID_NDK_HOME=$ANDROID_NDK_HOME"

# Cargo 构建参数 / Cargo build arguments
CARGO_FLAGS=""
PROFILE_DIR="debug"
if [ "$BUILD_TYPE" = "release" ]; then
    CARGO_FLAGS="--release"
    PROFILE_DIR="release"
fi

# 构建 / Build
echo ""
echo "[3/4] Building for Android targets..."

mkdir -p "$OUTPUT_DIR"

for target in "${TARGETS[@]}"; do
    echo ""
    echo "  Building for $target..."

    # 设置 ndk 版本 (Android API 21+ 推荐 / recommended)
    cargo ndk \
        --target "$target" \
        --platform 21 \
        build $CARGO_FLAGS

    echo "  [OK] $target done"
done

# 收集产物 / Collect artifacts
echo ""
echo "[4/4] Collecting build artifacts..."

for target in "${TARGETS[@]}"; do
    # 映射 target 到 Android ABI / Map target to Android ABI
    case "$target" in
        aarch64-linux-android)       ABI="arm64-v8a" ;;
        armv7-linux-androideabi)      ABI="armeabi-v7a" ;;
        x86_64-linux-android)        ABI="x86_64" ;;
        i686-linux-android)          ABI="x86" ;;
        *)                           ABI="unknown" ;;
    esac

    SRC="$PROJECT_DIR/target/$target/$PROFILE_DIR/libface_android.so"
    DST="$OUTPUT_DIR/$ABI/libface_android.so"

    if [ -f "$SRC" ]; then
        mkdir -p "$(dirname "$DST")"
        cp "$SRC" "$DST"
        echo "  [OK] $ABI/libbeauty_rs.so ($(du -h "$DST" | cut -f1))"
    else
        echo "  [WARN] $target binary not found at $SRC"
    fi
done

echo ""
echo "===================================================================="
echo "  Build complete!"
echo "  Output: $OUTPUT_DIR"
echo ""
echo "  To use in Android project, copy the .so files:"
echo "    cp -r $OUTPUT_DIR/* <android-project>/app/src/main/jniLibs/"
echo "===================================================================="