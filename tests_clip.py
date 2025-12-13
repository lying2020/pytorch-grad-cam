"""
CLIP模型可视化测试脚本
支持使用本地CLIP模型路径进行图像-文本匹配可视化

功能：
1. 将图像切分成可自定义的网格（默认7x7，可通过--patch-num参数调整）
2. 生成完整图片的热力图
3. 生成所有可能的组合patch热力图（从最小size到最大size）
4. 按类型合成大图并保存
5. 支持从JSON文件批量处理多张图片

组合patch生成规则：
- 默认网格大小：7x7
- 默认最小组合尺寸：4x4（patch_num - 3）
- 会生成从最小size到最大size的所有可能组合（例如：4x4, 4x5, 5x4, 5x5, 5x6, 6x5, 6x6等）
"""
import argparse
import os
import json
import glob
import shutil
import cv2
import numpy as np
import torch
from torch import nn
from pytorch_grad_cam import (
    GradCAM, FEM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
    AblationCAM, XGradCAM, EigenCAM, EigenGradCAM,
    LayerCAM, FullGrad, GradCAMElementWise, KPCA_CAM, ShapleyCAM,
    FinerCAM
)
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image, preprocess_image
)
from pytorch_grad_cam.utils.model_targets import RawScoresOutputTarget
from pytorch_grad_cam.ablation_layer import AblationLayerVit

try:
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    print("错误: transformers包未安装。请运行: pip install transformers")
    exit(1)

current_path = os.path.dirname(os.path.abspath(__file__))


def auto_detect_device():
    """
    自动检测可用的设备
    优先使用CUDA，如果不可用则使用CPU
    """
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"检测到CUDA可用，使用GPU加速: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("未检测到CUDA，使用CPU")
    return device


def get_args():
    parser = argparse.ArgumentParser(
        description='CLIP模型可视化 - 给定图像和文本，可视化模型关注与文本相关的区域'
    )
    parser.add_argument('--device', type=str, default=None,
                        help='Torch设备 (cpu, cuda, cuda:0等)。如果未指定，将自动检测')
    parser.add_argument('--image-dir', type=str,
                        default=os.path.join(current_path, 'images'),
                        help='输入图像目录')
    parser.add_argument('--image-name', type=str,
                        default='bird_flycatcher.jpg',
                        help='输入图像文件名')
    parser.add_argument('--image-text', type=str,
                        default='A sharp bird\'s beak',
                        help='输入图像文本描述（用于输出文件名）')
    parser.add_argument('--labels', type=str, nargs='+',
                        default=["bird", "animal"],
                        help='需要识别的文本标签列表')
    parser.add_argument('--clip-model-path', type=str,
                        default='/home/liying/Documents/clip-vit-large-patch14',
                        help='CLIP模型本地路径')
    parser.add_argument('--aug-smooth', action='store_true',
                        help='应用测试时数据增强来平滑CAM')
    parser.add_argument('--eigen-smooth', action='store_true',
                        help='通过取第一主成分来减少噪声')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=[
                            'gradcam', 'fem', 'hirescam', 'gradcam++',
                            'scorecam', 'xgradcam', 'ablationcam',
                            'eigencam', 'eigengradcam', 'layercam',
                            'fullgrad', 'gradcamelementwise', 'kpcacam',
                            'shapleycam', 'finercam'
                        ],
                        help='CAM方法')
    parser.add_argument('--output-dir', type=str,
                        default=os.path.join(current_path, 'output'),
                        help='输出目录')
    parser.add_argument('--json-file', type=str, default="images/image-01.json",
                        help='JSON元数据文件路径（如果指定，将批量处理JSON中的所有图片）')
    parser.add_argument('--patch-num', type=int, default=7,
                        help='将图像切分的网格大小（patch_num x patch_num），默认7（即7x7网格）')
    parser.add_argument('--min-patch-size', type=int, default=None,
                        help='组合patch的最小尺寸（默认值为patch_num-3）。例如patch_num=7时，最小size=4，会生成4x4, 4x5, 5x4, 5x5等组合')
    parser.add_argument('--target-retention-ratio', type=float, nargs='+', default=[0.4, 0.2, 0.2, 0.2],
                        help='合并热力图的目标保留比例（0-1之间）。可以是单个值（所有层级使用相同比例）或4个值（第一遍、第二遍、第三遍、第四遍分别使用）。默认[0.4, 0.2, 0.2, 0.15]')
    args = parser.parse_args()

    # 处理target-retention-ratio参数：支持单个值或4个值的列表
    if len(args.target_retention_ratio) == 1:
        # 单个值，所有层级使用相同比例
        args.target_retention_ratio = [args.target_retention_ratio[0]] * 4
    elif len(args.target_retention_ratio) == 4:
        # 4个值，分别用于第一遍、第二遍、第三遍、第四遍过滤
        args.target_retention_ratio = args.target_retention_ratio
    else:
        raise ValueError(f"target-retention-ratio 必须是1个值或4个值，当前提供 {len(args.target_retention_ratio)} 个值")

    # 验证所有值都在合理范围内
    for i, ratio in enumerate(args.target_retention_ratio):
        if not (0.05 <= ratio <= 0.95):
            raise ValueError(f"target-retention-ratio[{i}] 必须在0.05-0.95之间，当前值: {ratio}")

    # 计算最小patch size（如果未指定，使用默认值patch_num-3）
    if args.min_patch_size is None:
        args.min_patch_size = args.patch_num - 3
    if args.min_patch_size < 1:
        args.min_patch_size = 1
    if args.min_patch_size > args.patch_num:
        args.min_patch_size = args.patch_num

    # 自动检测设备（如果用户未指定）
    if args.device is None:
        args.device = auto_detect_device()
    else:
        print(f'使用指定设备: "{args.device}"')
        # 验证设备是否可用
        if args.device.startswith('cuda') and not torch.cuda.is_available():
            print(f"警告: 指定的设备 '{args.device}' 不可用，将使用CPU")
            args.device = 'cpu'

    return args


def reshape_transform(tensor, height=16, width=16):
    """
    CLIP ViT的reshape transform
    CLIP使用patch size 14，输入224x224，所以是16x16 (224/14=16)
    """
    # 移除class token (第一个token)
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))
    # 将通道维度移到第一维，像CNN一样
    result = result.transpose(2, 3).transpose(1, 2)
    return result


class CLIPImageClassifier(nn.Module):
    """
    CLIP模型包装类，用于图像-文本匹配可视化
    支持单个文本描述或多个标签
    """
    def __init__(self, text_description, model_path, labels=None):
        super(CLIPImageClassifier, self).__init__()
        print(f"正在从本地路径加载CLIP模型: {model_path}")
        self.clip = CLIPModel.from_pretrained(model_path)
        self.processor = CLIPProcessor.from_pretrained(model_path)
        self.text_description = text_description
        self.labels = labels if labels else [text_description]

    def forward(self, x):
        """
        前向传播
        输入: x - 图像tensor (batch_size, 3, H, W)
        输出: logits - 图像与文本的相似度分数 (batch_size, 1)
        """
        # 使用单个文本描述
        text_inputs = self.processor(
            text=[self.text_description],
            return_tensors="pt",
            padding=True
        )

        # 确保输入需要梯度（这对于梯度计算是必需的）
        if not x.requires_grad:
            x = x.requires_grad_(True)

        # 为了确保梯度能够正确传播，我们需要确保模型能够计算梯度
        # 即使模型在eval模式下，只要输入需要梯度，输出也应该需要梯度
        with torch.set_grad_enabled(True):
            outputs = self.clip(
                pixel_values=x,
                input_ids=text_inputs['input_ids'].to(self.clip.device),
                attention_mask=text_inputs['attention_mask'].to(self.clip.device)
            )

            # 返回logits_per_image，这是图像与文本的相似度分数
            logits_per_image = outputs.logits_per_image

        return logits_per_image


def split_image_into_grid(image, grid_size=(4, 4)):
    """
    将图像切分成网格
    参数:
        image: numpy array, shape (H, W, 3)
        grid_size: tuple, (rows, cols)
    返回:
        patches: list of numpy arrays, 每个是小图
        positions: list of tuples, 每个小图在原图中的位置 (row, col)
    """
    h, w = image.shape[:2]
    rows, cols = grid_size
    patch_h = h // rows
    patch_w = w // cols

    patches = []
    positions = []

    for i in range(rows):
        for j in range(cols):
            y1 = i * patch_h
            y2 = (i + 1) * patch_h if i < rows - 1 else h
            x1 = j * patch_w
            x2 = (j + 1) * patch_w if j < cols - 1 else w

            patch = image[y1:y2, x1:x2]
            patches.append(patch)
            positions.append((i, j))

    return patches, positions


def generate_combined_patches(patches, positions, grid_size=(7, 7), min_patch_size=4):
    """
    生成所有可能的组合patch，从最小size到最大size的所有组合

    参数:
        patches: list of numpy arrays, 原始的小图列表
        positions: list of tuples, 每个小图的位置 (row, col)
        grid_size: tuple, (rows, cols)
        min_patch_size: int, 组合patch的最小尺寸（例如4表示最小4x4）
    返回:
        combined_patches: list of numpy arrays, 组合后的patch
        combined_info: list of dict, 每个组合的信息 {'type': '4x4'/'4x5'/..., 'positions': [(r1,c1), ...], 'name': '...'}
    """
    rows, cols = grid_size
    combined_patches = []
    combined_info = []

    # 创建一个位置到索引的映射
    pos_to_idx = {pos: idx for idx, pos in enumerate(positions)}

    # 生成从最小size到最大size的所有可能组合
    max_size = min(rows, cols)
    for h in range(min_patch_size, max_size + 1):  # 高度从min_patch_size到max_size
        for w in range(min_patch_size, max_size + 1):  # 宽度从min_patch_size到max_size
            # 遍历所有可能的起始位置
            for start_i in range(rows - h + 1):
                for start_j in range(cols - w + 1):
                    # 收集这个区域的所有位置
                    pos_list = []
                    for i in range(start_i, start_i + h):
                        for j in range(start_j, start_j + w):
                            pos_list.append((i, j))

                    # 检查所有位置是否都存在
                    if all(pos in pos_to_idx for pos in pos_list):
                        # 获取所有patch
                        patch_list = [patches[pos_to_idx[pos]] for pos in pos_list]

                        # 按行拼接：先水平拼接每一行，再垂直拼接所有行
                        rows_patches = []
                        for row_idx in range(h):
                            row_start = row_idx * w
                            row_end = row_start + w
                            row_patches = patch_list[row_start:row_end]
                            row_combined = np.hstack(row_patches)
                            rows_patches.append(row_combined)

                        # 垂直拼接所有行
                        combined = np.vstack(rows_patches)
                        combined_patches.append(combined)

                        # 生成类型名称（例如 '4x4', '4x5', '5x4' 等）
                        type_name = f'{h}x{w}'
                        combined_info.append({
                            'type': type_name,
                            'positions': pos_list,
                            'name': f'{type_name}_{start_i}_{start_j}'
                        })

    return combined_patches, combined_info


def merge_heatmaps(heatmaps, positions, grid_size=(4, 4), original_shape=None):
    """
    将多个热力图拼接成一个大热力图
    参数:
        heatmaps: list of numpy arrays, 每个是小热力图
        positions: list of tuples, 每个热力图的位置 (row, col)
        grid_size: tuple, (rows, cols)
        original_shape: tuple, 原始图像尺寸 (H, W)
    返回:
        merged_heatmap: numpy array, 拼接后的大热力图
    """
    rows, cols = grid_size

    if original_shape is None:
        # 如果没有指定原始尺寸，使用第一个热力图的尺寸计算
        h, w = heatmaps[0].shape
        total_h = h * rows
        total_w = w * cols
    else:
        total_h, total_w = original_shape

    # 计算每个小热力图的尺寸
    patch_h = total_h // rows
    patch_w = total_w // cols

    merged_heatmap = np.zeros((total_h, total_w), dtype=np.float32)

    for heatmap, (i, j) in zip(heatmaps, positions):
        # 将热力图resize到对应的小图尺寸
        resized_heatmap = cv2.resize(heatmap, (patch_w, patch_h))

        y1 = i * patch_h
        y2 = y1 + patch_h
        x1 = j * patch_w
        x2 = x1 + patch_w

        # 确保不越界
        y2 = min(y2, total_h)
        x2 = min(x2, total_w)
        resized_heatmap = resized_heatmap[:y2-y1, :x2-x1]

        merged_heatmap[y1:y2, x1:x2] = resized_heatmap

    return merged_heatmap


def apply_adaptive_threshold(heatmap, target_retention_ratio=0.25, description=""):
    """
    应用自适应阈值过滤（基于目标保留比例）

    参数:
        heatmap: numpy array, 输入热力图
        target_retention_ratio: float, 目标保留比例（0-1之间）
        description: str, 描述信息（用于打印）
    返回:
        thresholded_heatmap: numpy array, 阈值处理后的热力图
        threshold_value: float, 使用的阈值
        actual_ratio: float, 实际保留比例
    """
    # 确保目标比例在合理范围内
    target_ratio = max(0.05, min(0.95, target_retention_ratio))

    # 将热力图的所有值排序
    sorted_values = np.sort(heatmap.flatten())
    # 计算对应的阈值位置（保留top target_ratio的比例）
    threshold_idx = int(len(sorted_values) * (1 - target_ratio))
    threshold_value = sorted_values[threshold_idx]

    # 应用阈值
    thresholded_heatmap = np.where(
        heatmap >= threshold_value,
        heatmap,
        0.0
    )

    # 计算实际保留比例
    actual_ratio = np.sum(thresholded_heatmap > 0) / thresholded_heatmap.size

    if description:
        mean_value = np.mean(heatmap)
        print(f"    {description}")
        print(f"      目标保留比例: {target_ratio*100:.1f}%")
        print(f"      自适应阈值: {threshold_value:.4f} (相对均值: {threshold_value/mean_value:.2f}x)")
        print(f"      实际保留区域占比: {actual_ratio*100:.2f}%")

    return thresholded_heatmap, threshold_value, actual_ratio


def merge_combined_heatmaps(heatmap_data_list, patch_type, grid_size=(7, 7), original_shape=None):
    """
    将组合patch的热力图合成大图
    支持任意大小的组合类型（例如4x4, 4x5, 5x4, 5x5等）
    对于重叠区域，使用加权平均策略：
    - 先计算每个子图的平均heatmap值作为权重
    - 平均heatmap值高的子图在重叠区域有更大的权重
    - 这样重要性更高的子图会对重叠区域有更大的影响

    参数:
        heatmap_data_list: list of dict, 每个dict包含'heatmap'和'positions'
        patch_type: str, 组合类型，例如'4x4', '4x5', '5x4'等
        grid_size: tuple, (rows, cols) 网格大小
        original_shape: tuple, 原始图像尺寸 (H, W)
    返回:
        merged: numpy array, 合成后的大热力图
    """
    rows, cols = grid_size
    if original_shape is None:
        total_h, total_w = original_shape
    else:
        total_h, total_w = original_shape

    # 解析patch_type，获取高度和宽度
    h_size, w_size = map(int, patch_type.split('x'))

    # 使用加权累加来处理重叠区域
    merged_weighted_sum = np.zeros((total_h, total_w), dtype=np.float32)
    merged_weight_sum = np.zeros((total_h, total_w), dtype=np.float32)

    patch_h = total_h // rows
    patch_w = total_w // cols

    # 第一步：计算每个子图的平均heatmap值作为权重
    patch_weights = []
    for data in heatmap_data_list:
        heatmap = data['heatmap']
        # 计算该子图的平均heatmap值作为权重
        mean_value = np.mean(heatmap)
        patch_weights.append(mean_value)

    # 如果所有权重都为0，则使用均匀权重
    if sum(patch_weights) == 0:
        patch_weights = [1.0] * len(patch_weights)
    else:
        # 归一化权重，使权重总和等于patch数量（保持与平均值方法类似的尺度）
        total_weight = sum(patch_weights)
        patch_weights = [w / total_weight * len(patch_weights) for w in patch_weights]

    # 第二步：使用权重进行加权累加
    for data, weight in zip(heatmap_data_list, patch_weights):
        heatmap = data['heatmap']
        positions = data['positions']

        # 获取起始位置（第一个patch的位置）
        row, col = positions[0]

        # 计算这个组合patch在原图中的位置
        y1 = row * patch_h
        y2 = min(y1 + patch_h * h_size, total_h)
        x1 = col * patch_w
        x2 = min(x1 + patch_w * w_size, total_w)

        # 将热力图resize到对应区域的大小
        resized = cv2.resize(heatmap, (x2 - x1, y2 - y1))

        # 加权累加：heatmap值乘以权重
        merged_weighted_sum[y1:y2, x1:x2] += resized * weight
        merged_weight_sum[y1:y2, x1:x2] += weight

    # 计算加权平均：对于重叠区域，使用加权平均
    merged = np.where(merged_weight_sum > 0, merged_weighted_sum / merged_weight_sum, 0)
    return merged


def load_json_metadata(json_path):
    """加载JSON元数据文件"""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON文件不存在: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    return metadata


def create_cam_object(model, target_layers, method, cam_algorithm, reshape_transform):
    """创建CAM对象"""
    if method == "ablationcam":
        cam = cam_algorithm(
            model=model,
            target_layers=target_layers,
            reshape_transform=reshape_transform,
            ablation_layer=AblationLayerVit()
        )
    else:
        cam = cam_algorithm(
            model=model,
            target_layers=target_layers,
            reshape_transform=reshape_transform
        )
    return cam


def process_full_image(model, cam, rgb_img_original, original_shape, args):
    """处理完整图片，生成热力图"""
    print(f"\n{'='*60}")
    print(f"第一步：为完整图片生成{args.method.upper()}热力图")
    print(f"{'='*60}")

    # 将完整图片resize到224x224（CLIP输入尺寸）
    rgb_img_full = cv2.resize(rgb_img_original.copy(), (224, 224))
    rgb_img_full = np.float32(rgb_img_full) / 255

    # 预处理完整图片，并设置requires_grad=True以确保梯度计算
    full_input_tensor = preprocess_image(
        rgb_img_full,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    ).to(args.device)
    full_input_tensor.requires_grad_(True)

    # 为完整图片生成CAM
    with cam:
        cam.batch_size = 32
        targets = [RawScoresOutputTarget()]

        full_grayscale_cam = cam(
            input_tensor=full_input_tensor,
            targets=targets,
            aug_smooth=args.aug_smooth,
            eigen_smooth=args.eigen_smooth
        )
        full_grayscale_cam = full_grayscale_cam[0, :]

        # 将热力图resize回原始图片尺寸
        full_grayscale_cam_resized = cv2.resize(
            full_grayscale_cam,
            (original_shape[1], original_shape[0])
        )

        # 阈值处理：计算均值，小于均值的部分清零，只保留关键位置
        full_heatmap_mean_value = np.mean(full_grayscale_cam_resized)
        print(f"  完整图片heatmap均值: {full_heatmap_mean_value:.4f}")
        full_grayscale_cam_thresholded = np.where(
            full_grayscale_cam_resized >= full_heatmap_mean_value,
            full_grayscale_cam_resized,
            0.0
        )
        print(f"  阈值处理后，保留区域占比: {np.sum(full_grayscale_cam_thresholded > 0) / full_grayscale_cam_thresholded.size * 100:.2f}%")

        # 生成完整图片的热力图可视化（使用阈值处理后的heatmap）
        full_heatmap_uint8 = np.uint8(255 * full_grayscale_cam_thresholded)
        full_heatmap_colored = cv2.applyColorMap(full_heatmap_uint8, cv2.COLORMAP_JET)

        # 生成完整图片的CAM叠加图（使用阈值处理后的heatmap）
        rgb_img_for_cam = np.float32(rgb_img_original) / 255
        full_cam_image = show_cam_on_image(rgb_img_for_cam, full_grayscale_cam_thresholded, use_rgb=True)
        full_cam_image = cv2.cvtColor(full_cam_image, cv2.COLOR_RGB2BGR)

    return full_grayscale_cam_thresholded, full_heatmap_colored, full_cam_image, full_heatmap_mean_value




def process_combined_patches(model, cam_algorithm, target_layers, patches, positions, grid_size, min_patch_size, args):
    """处理组合patches（从最小size到最大size的所有组合）"""
    print(f"\n{'='*60}")
    print(f"第三步：生成组合patch（最小尺寸{min_patch_size}x{min_patch_size}）并生成热力图")
    print(f"{'='*60}")

    # 生成所有组合patch
    print(f"\n正在生成组合patch...")
    combined_patches, combined_info = generate_combined_patches(patches, positions, grid_size, min_patch_size)
    print(f"已生成 {len(combined_patches)} 个组合patch:")
    type_counts = {}
    for info in combined_info:
        type_counts[info['type']] = type_counts.get(info['type'], 0) + 1
    # 按类型排序输出
    for patch_type in sorted(type_counts.keys(), key=lambda x: (int(x.split('x')[0]), int(x.split('x')[1]))):
        count = type_counts[patch_type]
        print(f"  - {patch_type}: {count} 个")

    # 为组合patch创建CAM对象
    cam_combined = create_cam_object(model, target_layers, args.method, cam_algorithm, reshape_transform)

    print(f"\n正在为每个组合patch生成{args.method.upper()}热力图...")
    combined_heatmaps_data = []

    with cam_combined:
        cam_combined.batch_size = 32

        for idx, (combined_patch, info) in enumerate(zip(combined_patches, combined_info)):
            print(f"  处理组合patch {info['name']} ({info['type']}) [{idx+1}/{len(combined_patches)}]...")

            # 将组合patch resize到224x224（CLIP输入尺寸）
            combined_patch_resized = cv2.resize(combined_patch, (224, 224))

            # 预处理，并设置requires_grad=True以确保梯度计算
            combined_tensor = preprocess_image(
                combined_patch_resized,
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            ).to(args.device)
            combined_tensor.requires_grad_(True)

            # 生成CAM
            targets = [RawScoresOutputTarget()]

            grayscale_cam = cam_combined(
                input_tensor=combined_tensor,
                targets=targets,
                aug_smooth=args.aug_smooth,
                eigen_smooth=args.eigen_smooth
            )
            grayscale_cam = grayscale_cam[0, :]

            # 将热力图resize回组合patch原始尺寸
            combined_h, combined_w = combined_patch.shape[:2]
            grayscale_cam_resized = cv2.resize(grayscale_cam, (combined_w, combined_h))

            # 第一遍过滤：对每个子图应用自适应阈值过滤（使用第一遍的比例）
            grayscale_cam_filtered, _, _ = apply_adaptive_threshold(
                grayscale_cam_resized,
                target_retention_ratio=args.target_retention_ratio[0],
                description=""
            )

            # 保存热力图数据用于后续合成（使用过滤后的热力图）
            combined_heatmaps_data.append({
                'heatmap': grayscale_cam_filtered,
                'info': info
            })

    return combined_heatmaps_data, type_counts


def merge_heatmaps_by_type(all_heatmaps, positions, combined_heatmaps_data, grid_size, original_shape,
                           full_heatmap_mean_value, rgb_img, image_output_dir, args):
    """
    按类型合成大图
    返回所有类型的热力图数据，用于后续合并
    """
    print(f"\n{'='*60}")
    print(f"第四步：按组合类型合成大图")
    print(f"{'='*60}")

    # 保存组合patch的热力图数据，按类型分类
    combined_heatmaps_by_type = {}

    # 组合类型的热力图
    for data in combined_heatmaps_data:
        patch_type = data['info']['type']
        if patch_type not in combined_heatmaps_by_type:
            combined_heatmaps_by_type[patch_type] = []
        combined_heatmaps_by_type[patch_type].append({
            'heatmap': data['heatmap'],
            'positions': data['info']['positions'],
            'name': data['info']['name']
        })

    # 保存所有类型的热力图数据，用于后续合并
    all_merged_heatmaps = {}

    # 为每种类型生成合成大图（按类型排序）
    print(f"\n正在按类型合成大图...")
    sorted_types = sorted(combined_heatmaps_by_type.keys(),
                         key=lambda x: (int(x.split('x')[0]), int(x.split('x')[1])))
    for patch_type in sorted_types:
        if len(combined_heatmaps_by_type[patch_type]) > 0:
            print(f"  合成 {patch_type} 类型的大图...")
            merged_type_heatmap = merge_combined_heatmaps(
                combined_heatmaps_by_type[patch_type],
                patch_type,
                grid_size=grid_size,
                original_shape=original_shape
            )

            # 第二遍过滤：对合成后的每个类型热力图应用自适应阈值过滤
            # 判断是否为较大的组合（高度或宽度 >= 4）
            h, w = map(int, patch_type.split('x'))
            if h >= 4 or w >= 4:
                merged_type_heatmap, _, _ = apply_adaptive_threshold(
                    merged_type_heatmap,
                    target_retention_ratio=args.target_retention_ratio[1],
                    description=f"    第二遍过滤 ({patch_type}类型)"
                )

            # 保存热力图数据用于后续合并
            all_merged_heatmaps[patch_type] = merged_type_heatmap.copy()

            # 生成可视化
            merged_type_heatmap_uint8 = np.uint8(255 * merged_type_heatmap)
            merged_type_heatmap_colored = cv2.applyColorMap(merged_type_heatmap_uint8, cv2.COLORMAP_JET)
            merged_type_cam_image = show_cam_on_image(rgb_img, merged_type_heatmap, use_rgb=True)
            merged_type_cam_image = cv2.cvtColor(merged_type_cam_image, cv2.COLOR_RGB2BGR)

            # 保存到 merged_gradcam 子文件夹（除了 all_combined 之外的所有 merged 文件）
            merged_gradcam_dir = os.path.join(image_output_dir, 'merged_gradcam')
            os.makedirs(merged_gradcam_dir, exist_ok=True)

            merged_type_heatmap_path = os.path.join(
                merged_gradcam_dir,
                f'merged_{args.method}_{patch_type}_heatmap.jpg'
            )
            merged_type_cam_path = os.path.join(
                merged_gradcam_dir,
                f'merged_{args.method}_{patch_type}_cam.jpg'
            )

            cv2.imwrite(merged_type_heatmap_path, merged_type_heatmap_colored)
            cv2.imwrite(merged_type_cam_path, merged_type_cam_image)
            print(f"    ✓ {patch_type} 热力图: {os.path.basename(merged_type_heatmap_path)}")
            print(f"    ✓ {patch_type} CAM叠加图: {os.path.basename(merged_type_cam_path)}")

    return all_merged_heatmaps, image_output_dir


def process_single_image(image_path, image_text, model, cam_algorithm, target_layers, args, base_output_dir=None):
    """
    处理单张图片的完整流程

    参数:
        image_path: 图片路径
        image_text: 语义描述文本
        model: CLIP模型
        cam_algorithm: CAM算法类
        target_layers: 目标层
        args: 参数对象
        base_output_dir: 基础输出目录（如果为None，则使用args.output_dir/{image_name}）
    """
    # 加载和预处理图像
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")

    print(f"\n{'#'*60}")
    print(f"处理图片: {os.path.basename(image_path)}")
    print(f"语义描述: {image_text}")
    print(f"{'#'*60}")

    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]  # BGR to RGB
    original_shape = rgb_img.shape[:2]
    rgb_img_original = rgb_img.copy()
    rgb_img = np.float32(rgb_img) / 255

    # 创建输出目录结构
    os.makedirs(args.output_dir, exist_ok=True)
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    if base_output_dir is None:
        image_output_dir = os.path.join(args.output_dir, image_name)
    else:
        image_output_dir = base_output_dir
    os.makedirs(image_output_dir, exist_ok=True)

    # 创建CAM对象
    cam = create_cam_object(model, target_layers, args.method, cam_algorithm, reshape_transform)

    # 第一步：处理完整图片
    full_grayscale_cam_thresholded, full_heatmap_colored, full_cam_image, full_heatmap_mean_value = \
        process_full_image(model, cam, rgb_img_original, original_shape, args)

    # 保存完整图片结果（保持原有命名）
    full_heatmap_path = os.path.join(image_output_dir, f'full_{args.method}_heatmap.jpg')
    full_cam_path = os.path.join(image_output_dir, f'full_{args.method}_cam.jpg')
    cv2.imwrite(full_heatmap_path, full_heatmap_colored)
    cv2.imwrite(full_cam_path, full_cam_image)
    print(f"  ✓ 完整图片热力图: {full_heatmap_path}")
    print(f"  ✓ 完整图片CAM叠加图: {full_cam_path}")

    # 同时保存为merged格式（与组合patch命名保持一致，使用实际的grid_size）
    grid_size_str = f"{args.patch_num}x{args.patch_num}"
    merged_full_heatmap_path = os.path.join(image_output_dir, f'merged_{args.method}_{grid_size_str}_heatmap.jpg')
    merged_full_cam_path = os.path.join(image_output_dir, f'merged_{args.method}_{grid_size_str}_cam.jpg')
    cv2.imwrite(merged_full_heatmap_path, full_heatmap_colored)
    cv2.imwrite(merged_full_cam_path, full_cam_image)
    print(f"  ✓ {grid_size_str}完整图片热力图: {merged_full_heatmap_path}")
    print(f"  ✓ {grid_size_str}完整图片CAM叠加图: {merged_full_cam_path}")

    # 第二步：切分图像为网格（用于生成组合patches）
    grid_size = (args.patch_num, args.patch_num)
    min_patch_size = args.min_patch_size
    print(f"\n{'='*60}")
    print(f"第二步：将图像切分成{args.patch_num}x{args.patch_num}网格（用于生成组合patches）")
    print(f"最小组合尺寸: {min_patch_size}x{min_patch_size}")
    print(f"{'='*60}")
    patches, positions = split_image_into_grid(rgb_img, grid_size)
    print(f"已切分成 {len(patches)} 个小图，用于生成组合patches")

    # 第三步：处理组合patches
    combined_output_dir = os.path.join(image_output_dir, 'combined_patches')
    os.makedirs(combined_output_dir, exist_ok=True)
    combined_heatmaps_data, type_counts = process_combined_patches(
        model, cam_algorithm, target_layers, patches, positions, grid_size, min_patch_size, args
    )

    # 保存组合patches结果
    for data in combined_heatmaps_data:
        info = data['info']
        heatmap = data['heatmap']
        # 找到对应的combined_patch（需要从patches重建）
        # 这里简化处理，只保存heatmap和cam
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(combined_output_dir, f"{info['name']}_heatmap.jpg"), heatmap_colored)

    # 第四步：按类型合成大图
    all_merged_heatmaps, _ = merge_heatmaps_by_type(
        None, positions, combined_heatmaps_data, grid_size, original_shape,
        full_heatmap_mean_value, rgb_img, image_output_dir, args
    )

    # 第五步：合并所有类型的热力图（使用像素级加权平均，类似自注意力机制）
    if len(all_merged_heatmaps) > 0:
        print(f"\n{'='*60}")
        print(f"第五步：合并所有组合类型的热力图（像素级加权平均，类似自注意力机制）")
        print(f"{'='*60}")

        # 获取第一个热力图的形状
        first_heatmap = list(all_merged_heatmaps.values())[0]
        h, w = first_heatmap.shape
        combined_all_heatmap = np.zeros((h, w), dtype=np.float32)

        # 对每个像素位置进行加权平均（类似自注意力机制）
        # 使用向量化操作提高效率
        # 将所有热力图堆叠成一个3D数组 (num_types, h, w)
        heatmap_list = list(all_merged_heatmaps.values())
        heatmap_stack = np.stack(heatmap_list, axis=0)  # shape: (num_types, h, w)
        num_types = len(heatmap_list)

        # 直接使用像素值作为权重的基础（类似自注意力机制）
        # 对每个像素位置，在所有类型间进行softmax归一化
        # 将heatmap_stack reshape为 (num_types, h*w)，然后对第一个维度进行softmax
        heatmap_flat = heatmap_stack.reshape(num_types, -1)  # (num_types, h*w)

        # Softmax归一化：exp(x) / sum(exp(x))
        # 减去最大值避免数值溢出
        heatmap_flat_exp = np.exp(heatmap_flat - np.max(heatmap_flat, axis=0, keepdims=True))
        weight_flat_normalized = heatmap_flat_exp / (np.sum(heatmap_flat_exp, axis=0, keepdims=True) + 1e-10)
        weight_stack_normalized = weight_flat_normalized.reshape(num_types, h, w)

        # 使用归一化后的权重进行加权平均
        for idx, heatmap in enumerate(heatmap_list):
            combined_all_heatmap += heatmap * weight_stack_normalized[idx]

        # 计算统计信息
        combined_mean_value = np.mean(combined_all_heatmap)
        print(f"  合并后热力图均值: {combined_mean_value:.4f}")

        # 第三遍过滤：对合并后的综合热力图应用自适应阈值过滤
        combined_all_heatmap_thresholded, threshold_value, actual_ratio = apply_adaptive_threshold(
            combined_all_heatmap,
            target_retention_ratio=args.target_retention_ratio[2],
            description="  第三遍过滤（所有类型合并后）"
        )

        # 生成可视化
        combined_all_heatmap_uint8 = np.uint8(255 * combined_all_heatmap_thresholded)
        combined_all_heatmap_colored = cv2.applyColorMap(combined_all_heatmap_uint8, cv2.COLORMAP_JET)
        combined_all_cam_image = show_cam_on_image(rgb_img, combined_all_heatmap_thresholded, use_rgb=True)
        combined_all_cam_image = cv2.cvtColor(combined_all_cam_image, cv2.COLOR_RGB2BGR)

        # 保存
        combined_all_heatmap_path = os.path.join(
            image_output_dir,
            f'merged_{args.method}_all_combined_heatmap.jpg'
        )
        combined_all_cam_path = os.path.join(
            image_output_dir,
            f'merged_{args.method}_all_combined_cam.jpg'
        )

        cv2.imwrite(combined_all_heatmap_path, combined_all_heatmap_colored)
        cv2.imwrite(combined_all_cam_path, combined_all_cam_image)
        print(f"  ✓ 所有类型合并热力图: {os.path.basename(combined_all_heatmap_path)}")
        print(f"  ✓ 所有类型合并CAM叠加图: {os.path.basename(combined_all_cam_path)}")

        # 保存原始图像
        original_output_path = os.path.join(image_output_dir, f'original.jpg')
        cv2.imwrite(original_output_path, cv2.cvtColor((rgb_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

        print(f"\n{'='*60}")
        print(f"完成！所有可视化结果已保存到: {image_output_dir}")
        print(f"{'='*60}")

        # 返回合并后的热力图（在应用阈值之前），用于后续合并所有语义
        return image_output_dir, combined_all_heatmap, rgb_img
    else:
        # 保存原始图像
        original_output_path = os.path.join(image_output_dir, f'original.jpg')
        cv2.imwrite(original_output_path, cv2.cvtColor((rgb_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

        print(f"\n{'='*60}")
        print(f"完成！所有可视化结果已保存到: {image_output_dir}")
        print(f"{'='*60}")

        # 如果没有生成合并热力图，返回None
        return image_output_dir, None, rgb_img


if __name__ == '__main__':
    """
    使用示例:
    # 处理单张图片
    python tests_clip.py --image-name bird_flycatcher.jpg --image-text "bird flycatcher"

    # 从JSON文件批量处理
    python tests_clip.py --json-file images/image_metadata.json --method gradcam

    # 指定CLIP模型路径
    python tests_clip.py --json-file images/image_metadata.json --clip-model-path /path/to/clip
    """
    args = get_args()

    # 定义所有可用的CAM方法
    methods = {
        "gradcam": GradCAM,
        "hirescam": HiResCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad,
        "fem": FEM,
        "gradcamelementwise": GradCAMElementWise,
        'kpcacam': KPCA_CAM,
        'shapleycam': ShapleyCAM,
        'finercam': FinerCAM
    }

    if args.method not in methods:
        raise ValueError(f"不支持的方法: {args.method}。可选: {list(methods.keys())}")

    # 检查模型路径是否存在
    if not os.path.exists(args.clip_model_path):
        raise FileNotFoundError(
            f"CLIP模型路径不存在: {args.clip_model_path}\n"
            f"请检查路径是否正确，或使用 --clip-model-path 指定正确的路径"
        )

    cam_algorithm = methods[args.method]

    # 如果指定了JSON文件，批量处理
    if args.json_file:
        print(f"\n{'='*60}")
        print(f"批量处理模式：从JSON文件读取图片列表")
        print(f"{'='*60}")

        # 加载JSON元数据
        metadata = load_json_metadata(args.json_file)
        images_list = metadata.get('images', [])

        if len(images_list) == 0:
            print("警告: JSON文件中没有图片数据")
            exit(1)

        print(f"找到 {len(images_list)} 张图片，开始批量处理...")

        # 为每张图片处理
        for idx, image_data in enumerate(images_list, 1):
            image_path = image_data.get('path') or os.path.join(
                metadata.get('images_dir', args.image_dir),
                image_data['filename']
            )

            # 获取图片名称（不含扩展名）
            image_name = os.path.splitext(image_data['filename'])[0]
            base_image_output_dir = os.path.join(args.output_dir, image_name)

            print(f"\n{'='*60}")
            print(f"处理进度: [{idx}/{len(images_list)}] - {image_data['filename']}")
            print(f"{'='*60}")

            # 第一步：处理 text_description（整体语义）
            text_description = image_data.get('text_description', '')
            if text_description:
                print(f"\n{'='*60}")
                print(f"步骤1: 处理整体语义描述 (text_description)")
                print(f"语义: {text_description}")
                print(f"{'='*60}")

                # 创建模型（使用text_description）
                model = CLIPImageClassifier(
                    text_description=text_description,
                    model_path=args.clip_model_path,
                    labels=args.labels
                ).to(torch.device(args.device))
                model.eval()
                target_layers = [model.clip.vision_model.encoder.layers[-1].layer_norm1]

                try:
                    # 处理单张图片，结果保存到 base_image_output_dir
                    _, _, _ = process_single_image(image_path, text_description, model, cam_algorithm, target_layers, args, base_output_dir=base_image_output_dir)

                except Exception as e:
                    print(f"错误: 处理图片 {image_data['filename']} 的 text_description 时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            # 第二步：处理 semantic 数组中的每个子语义
            semantic_list = image_data.get('semantic', [])
            if isinstance(semantic_list, str):
                # 如果semantic是字符串，转换为列表
                semantic_list = [semantic_list]

            # 收集所有子语义的合并热力图
            all_semantic_heatmaps = []
            all_semantic_rgb_imgs = []

            if semantic_list and len(semantic_list) > 0:
                print(f"\n{'='*60}")
                print(f"步骤2: 处理子语义描述 (semantic数组，共{len(semantic_list)}个)")
                print(f"{'='*60}")

                for semantic_idx, semantic_text in enumerate(semantic_list):
                    if not semantic_text or not semantic_text.strip():
                        continue

                    print(f"\n{'='*60}")
                    print(f"处理子语义 [{semantic_idx+1}/{len(semantic_list)}]: {semantic_text}")
                    print(f"{'='*60}")

                    # 创建子语义输出目录
                    semantic_output_dir = os.path.join(base_image_output_dir, f'semantic_{semantic_idx}')

                    # 创建模型（使用当前子语义）
                    model = CLIPImageClassifier(
                        text_description=semantic_text,
                        model_path=args.clip_model_path,
                        labels=args.labels
                    ).to(torch.device(args.device))
                    model.eval()
                    target_layers = [model.clip.vision_model.encoder.layers[-1].layer_norm1]

                    try:
                        # 处理单张图片，结果保存到 semantic_output_dir
                        _, semantic_heatmap, semantic_rgb_img = process_single_image(
                            image_path, semantic_text, model, cam_algorithm, target_layers, args, base_output_dir=semantic_output_dir
                        )

                        # 收集合并后的热力图（如果存在）
                        if semantic_heatmap is not None:
                            all_semantic_heatmaps.append(semantic_heatmap)
                            all_semantic_rgb_imgs.append(semantic_rgb_img)

                        # 在子语义文件夹中创建txt文件保存语义信息
                        semantic_txt_path = os.path.join(semantic_output_dir, 'semantic_info.txt')
                        with open(semantic_txt_path, 'w', encoding='utf-8') as f:
                            f.write(f"Semantic Index: {semantic_idx}\n")
                            f.write(f"Semantic Text: {semantic_text}\n")
                        print(f"  ✓ 语义信息已保存到: semantic_info.txt")

                    except Exception as e:
                        print(f"错误: 处理图片 {image_data['filename']} 的子语义 {semantic_idx} ({semantic_text}) 时出错: {e}")
                        import traceback
                        traceback.print_exc()
                        continue

            # 步骤3：合并所有子语义的结果
            if len(all_semantic_heatmaps) > 0:
                print(f"\n{'='*60}")
                print(f"步骤3: 合并所有子语义的热力图（像素级加权平均）")
                print(f"{'='*60}")

                # 获取第一个热力图的形状
                first_heatmap = all_semantic_heatmaps[0]
                h, w = first_heatmap.shape
                combined_all_semantic_heatmap = np.zeros((h, w), dtype=np.float32)

                # 使用像素级加权平均合并所有子语义的热力图
                heatmap_stack = np.stack(all_semantic_heatmaps, axis=0)  # shape: (num_semantics, h, w)
                num_semantics = len(all_semantic_heatmaps)

                # 对每个像素位置，在所有语义间进行softmax归一化
                heatmap_flat = heatmap_stack.reshape(num_semantics, -1)  # (num_semantics, h*w)
                heatmap_flat_exp = np.exp(heatmap_flat - np.max(heatmap_flat, axis=0, keepdims=True))
                weight_flat_normalized = heatmap_flat_exp / (np.sum(heatmap_flat_exp, axis=0, keepdims=True) + 1e-10)
                weight_stack_normalized = weight_flat_normalized.reshape(num_semantics, h, w)

                # 使用归一化后的权重进行加权平均
                for idx, heatmap in enumerate(all_semantic_heatmaps):
                    combined_all_semantic_heatmap += heatmap * weight_stack_normalized[idx]

                # 计算统计信息
                combined_semantic_mean_value = np.mean(combined_all_semantic_heatmap)
                print(f"  合并后热力图均值: {combined_semantic_mean_value:.4f}")

                # 第四遍过滤：对合并后的所有语义热力图应用自适应阈值过滤
                combined_all_semantic_heatmap_thresholded, threshold_value, actual_ratio = apply_adaptive_threshold(
                    combined_all_semantic_heatmap,
                    target_retention_ratio=args.target_retention_ratio[3] if len(args.target_retention_ratio) > 3 else args.target_retention_ratio[2],
                    description="  第四遍过滤（所有语义合并后）"
                )

                # 使用第一个RGB图像进行可视化（所有语义应该使用同一张原图）
                rgb_img_for_combined = all_semantic_rgb_imgs[0] if len(all_semantic_rgb_imgs) > 0 else None

                if rgb_img_for_combined is not None:
                    # 生成可视化
                    combined_all_semantic_heatmap_uint8 = np.uint8(255 * combined_all_semantic_heatmap_thresholded)
                    combined_all_semantic_heatmap_colored = cv2.applyColorMap(combined_all_semantic_heatmap_uint8, cv2.COLORMAP_JET)
                    combined_all_semantic_cam_image = show_cam_on_image(rgb_img_for_combined, combined_all_semantic_heatmap_thresholded, use_rgb=True)
                    combined_all_semantic_cam_image = cv2.cvtColor(combined_all_semantic_cam_image, cv2.COLOR_RGB2BGR)

                    # 保存到主目录
                    combined_all_semantic_heatmap_path = os.path.join(
                        base_image_output_dir,
                        f'merged_{args.method}_all_semantics_combined_heatmap.jpg'
                    )
                    combined_all_semantic_cam_path = os.path.join(
                        base_image_output_dir,
                        f'merged_{args.method}_all_semantics_combined_cam.jpg'
                    )

                    cv2.imwrite(combined_all_semantic_heatmap_path, combined_all_semantic_heatmap_colored)
                    cv2.imwrite(combined_all_semantic_cam_path, combined_all_semantic_cam_image)
                    print(f"  ✓ 所有语义合并热力图: {os.path.basename(combined_all_semantic_heatmap_path)}")
                    print(f"  ✓ 所有语义合并CAM叠加图: {os.path.basename(combined_all_semantic_cam_path)}")

            # 步骤4：创建 summaries 文件夹并拷贝相关文件
            print(f"\n{'='*60}")
            print(f"步骤4: 创建 summaries 文件夹并整理结果")
            print(f"{'='*60}")

            summaries_dir = os.path.join(base_image_output_dir, 'summaries')
            os.makedirs(summaries_dir, exist_ok=True)

            # 拷贝 text_description 生成的结果和原图
            if text_description:
                # 拷贝原图
                original_path = os.path.join(base_image_output_dir, 'original.jpg')
                if os.path.exists(original_path):
                    summaries_original_path = os.path.join(summaries_dir, 'original.jpg')
                    shutil.copy2(original_path, summaries_original_path)
                    print(f"  ✓ 拷贝原图到 summaries/")

                # 拷贝 full_gradcam 文件
                full_heatmap_path = os.path.join(base_image_output_dir, f'full_{args.method}_heatmap.jpg')
                full_cam_path = os.path.join(base_image_output_dir, f'full_{args.method}_cam.jpg')
                if os.path.exists(full_heatmap_path):
                    summaries_full_heatmap = os.path.join(summaries_dir, f'full_{args.method}_heatmap.jpg')
                    shutil.copy2(full_heatmap_path, summaries_full_heatmap)
                    print(f"  ✓ 拷贝 full_{args.method}_heatmap.jpg 到 summaries/")
                if os.path.exists(full_cam_path):
                    summaries_full_cam = os.path.join(summaries_dir, f'full_{args.method}_cam.jpg')
                    shutil.copy2(full_cam_path, summaries_full_cam)
                    print(f"  ✓ 拷贝 full_{args.method}_cam.jpg 到 summaries/")

                # 拷贝 merged_gradcam_all_combined 文件
                all_combined_heatmap = os.path.join(base_image_output_dir, f'merged_{args.method}_all_combined_heatmap.jpg')
                all_combined_cam = os.path.join(base_image_output_dir, f'merged_{args.method}_all_combined_cam.jpg')
                if os.path.exists(all_combined_heatmap):
                    summaries_all_combined_heatmap = os.path.join(summaries_dir, f'merged_{args.method}_all_combined_heatmap.jpg')
                    shutil.copy2(all_combined_heatmap, summaries_all_combined_heatmap)
                    print(f"  ✓ 拷贝 merged_{args.method}_all_combined_heatmap.jpg 到 summaries/")
                if os.path.exists(all_combined_cam):
                    summaries_all_combined_cam = os.path.join(summaries_dir, f'merged_{args.method}_all_combined_cam.jpg')
                    shutil.copy2(all_combined_cam, summaries_all_combined_cam)
                    print(f"  ✓ 拷贝 merged_{args.method}_all_combined_cam.jpg 到 summaries/")

            # 拷贝每个子语义的 full_gradcam 和 merged_gradcam_all_combined 文件（加上语义文件夹名字的后缀）
            if semantic_list and len(semantic_list) > 0:
                for semantic_idx, semantic_text in enumerate(semantic_list):
                    if not semantic_text or not semantic_text.strip():
                        continue

                    semantic_output_dir = os.path.join(base_image_output_dir, f'semantic_{semantic_idx}')
                    suffix = f'_semantic_{semantic_idx}'

                    # 拷贝 full_gradcam 文件
                    semantic_full_heatmap = os.path.join(semantic_output_dir, f'full_{args.method}_heatmap.jpg')
                    semantic_full_cam = os.path.join(semantic_output_dir, f'full_{args.method}_cam.jpg')

                    if os.path.exists(semantic_full_heatmap):
                        summaries_semantic_full_heatmap = os.path.join(
                            summaries_dir,
                            f'full_{args.method}_heatmap{suffix}.jpg'
                        )
                        shutil.copy2(semantic_full_heatmap, summaries_semantic_full_heatmap)
                        print(f"  ✓ 拷贝 full_{args.method}_heatmap{suffix}.jpg 到 summaries/")

                    if os.path.exists(semantic_full_cam):
                        summaries_semantic_full_cam = os.path.join(
                            summaries_dir,
                            f'full_{args.method}_cam{suffix}.jpg'
                        )
                        shutil.copy2(semantic_full_cam, summaries_semantic_full_cam)
                        print(f"  ✓ 拷贝 full_{args.method}_cam{suffix}.jpg 到 summaries/")

                    # 拷贝 merged_gradcam_all_combined 文件
                    semantic_all_combined_heatmap = os.path.join(semantic_output_dir, f'merged_{args.method}_all_combined_heatmap.jpg')
                    semantic_all_combined_cam = os.path.join(semantic_output_dir, f'merged_{args.method}_all_combined_cam.jpg')

                    if os.path.exists(semantic_all_combined_heatmap):
                        summaries_semantic_all_combined_heatmap = os.path.join(
                            summaries_dir,
                            f'merged_{args.method}_all_combined_heatmap{suffix}.jpg'
                        )
                        shutil.copy2(semantic_all_combined_heatmap, summaries_semantic_all_combined_heatmap)
                        print(f"  ✓ 拷贝 merged_{args.method}_all_combined_heatmap{suffix}.jpg 到 summaries/")

                    if os.path.exists(semantic_all_combined_cam):
                        summaries_semantic_all_combined_cam = os.path.join(
                            summaries_dir,
                            f'merged_{args.method}_all_combined_cam{suffix}.jpg'
                        )
                        shutil.copy2(semantic_all_combined_cam, summaries_semantic_all_combined_cam)
                        print(f"  ✓ 拷贝 merged_{args.method}_all_combined_cam{suffix}.jpg 到 summaries/")

            # 拷贝合并所有语义的结果
            all_semantics_combined_heatmap = os.path.join(base_image_output_dir, f'merged_{args.method}_all_semantics_combined_heatmap.jpg')
            all_semantics_combined_cam = os.path.join(base_image_output_dir, f'merged_{args.method}_all_semantics_combined_cam.jpg')
            if os.path.exists(all_semantics_combined_heatmap):
                summaries_all_semantics_heatmap = os.path.join(summaries_dir, f'merged_{args.method}_all_semantics_combined_heatmap.jpg')
                shutil.copy2(all_semantics_combined_heatmap, summaries_all_semantics_heatmap)
                print(f"  ✓ 拷贝 merged_{args.method}_all_semantics_combined_heatmap.jpg 到 summaries/")
            if os.path.exists(all_semantics_combined_cam):
                summaries_all_semantics_cam = os.path.join(summaries_dir, f'merged_{args.method}_all_semantics_combined_cam.jpg')
                shutil.copy2(all_semantics_combined_cam, summaries_all_semantics_cam)
                print(f"  ✓ 拷贝 merged_{args.method}_all_semantics_combined_cam.jpg 到 summaries/")

        print(f"\n{'='*60}")
        print(f"批量处理完成！共处理 {len(images_list)} 张图片")
        print(f"{'='*60}")

    else:
        # 单张图片处理模式（原有逻辑）
        print(f"\n正在加载CLIP模型...")
        print(f"使用文本描述: {args.image_text}")
        model = CLIPImageClassifier(
            text_description=args.image_text,
            model_path=args.clip_model_path,
            labels=args.labels
        ).to(torch.device(args.device))
        model.eval()

        # 选择目标层 - CLIP ViT的最后一层layer norm
        target_layers = [model.clip.vision_model.encoder.layers[-1].layer_norm1]
        print(f"目标层: {target_layers[0]}")

        # 处理单张图片
        image_path = os.path.join(args.image_dir, args.image_name)
        process_single_image(image_path, args.image_text, model, cam_algorithm, target_layers, args)
