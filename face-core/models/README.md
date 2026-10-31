# beauty-rs: Cross-Platform Face Beauty Engine
#
# 将模型文件（*.onnx）放在此目录中，编译时通过 include_bytes! 内嵌到二进制。
# Place model files (*.onnx) in this directory; they get embedded into the
# binary at compile time via include_bytes!.
#
# 必需文件 / Required files:
#   blazeface_128.onnx   — BlazeFace short-range face detector (128x128, ~460KB)
#   facemesh_468.onnx    — MediaPipe FaceMesh V1 468-point 3D mesh (192x192)
#   bisenet_r18.onnx     — BiSeNet ResNet18 face parsing 19-class (512x512)
#
# 下载方法 / How to download:
#   python scripts/download_models.py