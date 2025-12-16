#!/bin/bash

# 源路径和目标路径
SOURCE_DIR="/home/liying/Documents/dataset/VLMDataset/fgvc_aircraft/images"
TARGET_DIR="/home/liying/Desktop/EDPL-AAAI-2025/pytorch-grad-cam/outputs/fgvc_aircraft"

# 创建目标目录（如果不存在）
mkdir -p "$TARGET_DIR"

# 文件名列表
files=(
    "1025794"
    "1340192"
    "0056978"
    "0698580"
    "0450014"
    "1042824"
    "0894380"
    "1427680"
    "0817494"
    "0716386"
    "0951982"
    "0731614"
    "0582363"
    "1082409"
    "2031775"
    "0950991"
    "0869722"
    "0979376"
    "1002439"
    "0864665"
    "1207591"
    "0582372"
    "0729223"
    "1319365"
    "0548719"
    "0577855"
    "1423583"
    "1187431"
    "0610657"
    "0869742"
    "0687610"
    "1042021"
    "0482761"
    "0064933"
)

# 常见的图片扩展名
extensions=("jpg" "jpeg" "png" "JPG" "JPEG" "PNG")

# 计数器
copied=0
not_found=0

echo "开始拷贝图片..."
echo "源路径: $SOURCE_DIR"
echo "目标路径: $TARGET_DIR"
echo ""

# 遍历每个文件名
for filename in "${files[@]}"; do
    found=false

    # 尝试不同的扩展名
    for ext in "${extensions[@]}"; do
        source_file="$SOURCE_DIR/${filename}.${ext}"
        if [ -f "$source_file" ]; then
            cp "$source_file" "$TARGET_DIR/"
            echo "✓ 已拷贝: ${filename}.${ext}"
            ((copied++))
            found=true
            break
        fi
    done

    # 如果所有扩展名都找不到
    if [ "$found" = false ]; then
        echo "✗ 未找到: $filename (尝试了所有扩展名)"
        ((not_found++))
    fi
done

echo ""
echo "拷贝完成!"
echo "成功拷贝: $copied 个文件"
echo "未找到: $not_found 个文件"
