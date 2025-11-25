#!/bin/bash
# ===============================================================
# Hailo 单帧推理 C++ 示例自动构建与运行脚本
# 适配环境: Firefly RK3588J + Ubuntu + HailoRT SDK + OpenCV
# ===============================================================

# 1. 基本配置
PROJECT_NAME="single_infer"
SRC_FILE="single_infer.cpp"
BUILD_DIR="./build"
HEF_PATH="../dncnn_80ep_l9_4split_16pad.hef"
INPUT_IMAGE="20250113_0000.jpg"
OUTPUT_IMAGE="output.jpg"

# 2. 路径检查
if [ ! -f "$SRC_FILE" ]; then
    echo "❌ 源文件未找到: $SRC_FILE"
    exit 1
fi

if [ ! -f "$HEF_PATH" ]; then
    echo "❌ HEF 模型文件未找到: $HEF_PATH"
    exit 1
fi

if [ ! -f "$INPUT_IMAGE" ]; then
    echo "❌ 输入图像未找到: $INPUT_IMAGE"
    exit 1
fi

# 3. 环境变量（如 HailoRT 未全局安装时需手动指定）
export HAILORT_DIR="../../hailort/hailort"
export OpenCV_DIR="/usr/lib/aarch64-linux-gnu/cmake/opencv4"

if [ ! -d "$HAILORT_DIR" ]; then
    echo "⚠️  未找到 HailoRT SDK，请检查路径: $HAILORT_DIR"
fi

# 4. 创建构建目录
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR" || exit

# 5. 生成 CMake 构建系统
echo "🚀 运行 CMake 配置..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DHAILORT_DIR="$HAILORT_DIR"

# 6. 编译
echo "⚙️  开始编译..."
make -j$(nproc)
if [ $? -ne 0 ]; then
    echo "❌ 编译失败"
    exit 1
fi

# 7. 运行推理程序
echo "🎬 开始推理..."
./${PROJECT_NAME}
if [ $? -eq 0 ]; then
    echo "✅ 推理完成，输出图像位于: $OUTPUT_IMAGE"
else
    echo "❌ 推理运行失败"
    exit 1
fi

# 8. 自动查看输出结果
if command -v feh >/dev/null 2>&1; then
    echo "🖼️  打开结果预览..."
    feh "$OUTPUT_IMAGE" &
else
    echo "ℹ️  可使用以下命令查看结果:"
    echo "    feh $OUTPUT_IMAGE"
fi
