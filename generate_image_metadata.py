"""
生成images文件夹下所有图片的元数据JSON文件
包含图片名称和对应的语义信息
"""
import os
import json
import cv2
from pathlib import Path

current_path = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(current_path, 'images')
output_json_path = os.path.join(images_dir, 'image_metadata.json')


def extract_semantic_from_filename(filename):
    """
    从文件名提取语义信息
    例如:
    - bird_flycatcher.jpg -> "bird flycatcher"
    - car.png -> "car"
    - dog_pug_39.jpg -> "dog pug"
    - 000000039769.jpg -> "000000039769" (保持原样，可能是ID)
    """
    # 移除扩展名
    name_without_ext = os.path.splitext(filename)[0]

    # 将下划线替换为空格
    semantic = name_without_ext.replace('_', ' ')

    # 移除数字后缀（如 "dog_pug_39" -> "dog pug"）
    # 但保留有意义的数字（如 "000000039769" 保持原样）
    parts = semantic.split()
    if len(parts) > 1:
        # 如果最后一部分是纯数字且长度较短（可能是编号），尝试移除
        if parts[-1].isdigit() and len(parts[-1]) <= 3:
            semantic = ' '.join(parts[:-1])

    # 如果结果为空，使用原文件名（去掉扩展名）
    if not semantic.strip():
        semantic = name_without_ext

    return semantic


def get_image_info(image_path):
    """
    获取图片的基本信息
    """
    try:
        img = cv2.imread(image_path)
        if img is not None:
            h, w = img.shape[:2]
            return {
                'width': int(w),
                'height': int(h),
                'channels': int(img.shape[2]) if len(img.shape) > 2 else 1
            }
    except Exception as e:
        print(f"警告: 无法读取图片 {image_path}: {e}")
    return None


def generate_metadata():
    """
    扫描images文件夹，生成元数据JSON文件
    """
    if not os.path.exists(images_dir):
        print(f"错误: images文件夹不存在: {images_dir}")
        return

    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif', '.webp'}

    # 扫描所有图片文件
    image_files = []
    for filename in os.listdir(images_dir):
        file_path = os.path.join(images_dir, filename)
        if os.path.isfile(file_path):
            ext = os.path.splitext(filename)[1].lower()
            if ext in image_extensions:
                image_files.append(filename)

    if len(image_files) == 0:
        print(f"警告: 在 {images_dir} 中未找到图片文件")
        return

    # 生成元数据
    metadata = {
        'images_dir': images_dir,
        'total_images': len(image_files),
        'images': []
    }

    print(f"正在扫描 {images_dir} 文件夹...")
    print(f"找到 {len(image_files)} 个图片文件")

    for idx, filename in enumerate(sorted(image_files), 1):
        print(f"  处理 [{idx}/{len(image_files)}]: {filename}")

        image_path = os.path.join(images_dir, filename)
        semantic_info = extract_semantic_from_filename(filename)
        image_info = get_image_info(image_path)

        image_data = {
            'filename': filename,
            'semantic': semantic_info,
            'path': image_path
        }

        if image_info:
            image_data.update(image_info)

        metadata['images'].append(image_data)

    # 保存JSON文件
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 元数据已保存到: {output_json_path}")
    print(f"  包含 {len(image_files)} 个图片的元数据")

    return metadata


if __name__ == '__main__':
    print("="*60)
    print("生成图片元数据JSON文件")
    print("="*60)
    metadata = generate_metadata()

    if metadata:
        print("\n示例数据:")
        if len(metadata['images']) > 0:
            print(json.dumps(metadata['images'][0], ensure_ascii=False, indent=2))
