"""
CLIP模型可视化测试脚本
支持使用本地CLIP模型路径进行图像-文本匹配可视化

功能：
1. 将图像切分成4x4网格（16个小图）
2. 对每个小图，使用指定的文本描述生成热力图
3. 将所有小图的热力图拼接成完整的大热力图
4. 保存所有结果到output目录
"""
import argparse
import os
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
                        default='The bird beak for a bird',
                        help='输入图像文本描述（用于输出文件名）')
    parser.add_argument('--labels', type=str, nargs='+',
                        default=["a bird", "a cat", "a dog", "a car"],
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
    args = parser.parse_args()

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


if __name__ == '__main__':
    """
    使用示例:
    python tests_clip.py --image-name bird_flycatcher.jpg --labels "a bird" "a cat" "a dog"
    python tests_clip.py --clip-model-path /path/to/clip --method gradcam --aug-smooth
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

    # 加载CLIP模型
    print(f"\n正在加载CLIP模型...")
    print(f"使用文本描述: {args.image_text}")
    model = CLIPImageClassifier(
        text_description=args.image_text,
        model_path=args.clip_model_path,
        labels=args.labels
    ).to(torch.device(args.device))
    # 注意：不要设置为eval()模式，因为我们需要计算梯度
    # 但为了确保模型行为一致，我们可以在需要时临时设置为train模式
    model.eval()  # 先设置为eval
    # 但我们需要确保在计算梯度时，相关层能够保留梯度

    # 选择目标层 - CLIP ViT的最后一层layer norm
    target_layers = [model.clip.vision_model.encoder.layers[-1].layer_norm1]
    print(f"目标层: {target_layers[0]}")

    # 加载和预处理图像
    image_path = os.path.join(args.image_dir, args.image_name)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")

    print(f"正在加载图像: {image_path}")
    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]  # BGR to RGB
    original_shape = rgb_img.shape[:2]  # 保存原始尺寸
    rgb_img_original = rgb_img.copy()  # 保存原始图像用于完整图片处理
    rgb_img = np.float32(rgb_img) / 255

    # 创建输出目录结构
    os.makedirs(args.output_dir, exist_ok=True)
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # 创建以图片名命名的输出目录
    image_output_dir = os.path.join(args.output_dir, image_name)
    os.makedirs(image_output_dir, exist_ok=True)

    # 创建CAM对象
    cam_algorithm = methods[args.method]

    # 创建CAM对象
    if args.method == "ablationcam":
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

    # ========== 第一步：为完整图片生成热力图 ==========
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

        # 生成完整图片的热力图可视化
        full_heatmap_uint8 = np.uint8(255 * full_grayscale_cam_resized)
        full_heatmap_colored = cv2.applyColorMap(full_heatmap_uint8, cv2.COLORMAP_JET)

        # 生成完整图片的CAM叠加图（使用原始尺寸的图像）
        rgb_img_for_cam = np.float32(rgb_img_original) / 255
        full_cam_image = show_cam_on_image(rgb_img_for_cam, full_grayscale_cam_resized, use_rgb=True)
        full_cam_image = cv2.cvtColor(full_cam_image, cv2.COLOR_RGB2BGR)

        # 保存完整图片的结果到图片名目录下
        full_heatmap_path = os.path.join(
            image_output_dir,
            f'full_{args.method}_heatmap.jpg'
        )
        full_cam_path = os.path.join(
            image_output_dir,
            f'full_{args.method}_cam.jpg'
        )
        full_original_path = os.path.join(
            image_output_dir,
            f'full_original.jpg'
        )

        cv2.imwrite(full_heatmap_path, full_heatmap_colored)
        cv2.imwrite(full_cam_path, full_cam_image)
        cv2.imwrite(
            full_original_path,
            cv2.cvtColor((rgb_img_for_cam * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        )

        print(f"  ✓ 完整图片热力图: {full_heatmap_path}")
        print(f"  ✓ 完整图片CAM叠加图: {full_cam_path}")
        print(f"  ✓ 完整图片原始图像: {full_original_path}")

    # ========== 第二步：将图像切分成4x4网格并处理 ==========
    print(f"\n{'='*60}")
    print(f"第二步：将图像切分成4x4网格并生成热力图")
    print(f"{'='*60}")

    # 将图像切分成4x4网格
    grid_size = (4, 4)
    print(f"\n正在将图像切分成 {grid_size[0]}x{grid_size[1]} 网格...")
    patches, positions = split_image_into_grid(rgb_img, grid_size)
    print(f"已切分成 {len(patches)} 个小图")

    # 为每个小图生成热力图
    print(f"\n正在为每个小图生成{args.method.upper()}热力图...")
    all_heatmaps = []
    all_patch_heatmaps_colored = []
    all_patch_cam_images = []

    # 小图结果也保存在图片名目录下的patches子目录
    patch_output_dir = os.path.join(image_output_dir, 'patches')
    os.makedirs(patch_output_dir, exist_ok=True)

    # 为小图重新创建CAM对象，确保状态正确
    if args.method == "ablationcam":
        cam_patches = cam_algorithm(
            model=model,
            target_layers=target_layers,
            reshape_transform=reshape_transform,
            ablation_layer=AblationLayerVit()
        )
    else:
        cam_patches = cam_algorithm(
            model=model,
            target_layers=target_layers,
            reshape_transform=reshape_transform
        )

    with cam_patches:
        cam_patches.batch_size = 32

        for idx, (patch, (row, col)) in enumerate(zip(patches, positions)):
            print(f"  处理小图 ({row}, {col}) [{idx+1}/{len(patches)}]...")

            # 将小图resize到224x224（CLIP输入尺寸）
            patch_resized = cv2.resize(patch, (224, 224))

            # 预处理，并设置requires_grad=True以确保梯度计算
            patch_tensor = preprocess_image(
                patch_resized,
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            ).to(args.device)
            patch_tensor.requires_grad_(True)

            # 生成CAM - 使用RawScoresOutputTarget来直接使用logits
            # CLIP模型的输出是logits_per_image，表示图像与文本的相似度
            # 我们想要最大化这个相似度，所以直接使用原始输出
            targets = [RawScoresOutputTarget()]

            grayscale_cam = cam_patches(
                input_tensor=patch_tensor,
                targets=targets,
                aug_smooth=args.aug_smooth,
                eigen_smooth=args.eigen_smooth
            )
            grayscale_cam = grayscale_cam[0, :]

            # 将热力图resize回小图原始尺寸
            patch_h, patch_w = patch.shape[:2]
            grayscale_cam_resized = cv2.resize(grayscale_cam, (patch_w, patch_h))
            all_heatmaps.append(grayscale_cam_resized)

            # 生成小图的热力图可视化
            heatmap_uint8 = np.uint8(255 * grayscale_cam_resized)
            heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            all_patch_heatmaps_colored.append(heatmap_colored)

            # 生成小图的CAM叠加图
            patch_cam_image = show_cam_on_image(patch, grayscale_cam_resized, use_rgb=True)
            patch_cam_image = cv2.cvtColor(patch_cam_image, cv2.COLOR_RGB2BGR)
            all_patch_cam_images.append(patch_cam_image)

            # 保存每个小图的结果
            patch_heatmap_path = os.path.join(
                patch_output_dir,
                f'patch_{row}_{col}_heatmap.jpg'
            )
            patch_cam_path = os.path.join(
                patch_output_dir,
                f'patch_{row}_{col}_cam.jpg'
            )
            patch_original_path = os.path.join(
                patch_output_dir,
                f'patch_{row}_{col}_original.jpg'
            )

            cv2.imwrite(patch_heatmap_path, heatmap_colored)
            cv2.imwrite(patch_cam_path, patch_cam_image)
            cv2.imwrite(
                patch_original_path,
                cv2.cvtColor((patch * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            )

    # 拼接所有热力图
    print(f"\n正在拼接 {grid_size[0]}x{grid_size[1]} 个热力图...")
    merged_heatmap = merge_heatmaps(
        all_heatmaps,
        positions,
        grid_size=grid_size,
        original_shape=original_shape
    )

    # 生成拼接后的可视化
    merged_heatmap_uint8 = np.uint8(255 * merged_heatmap)
    merged_heatmap_colored = cv2.applyColorMap(merged_heatmap_uint8, cv2.COLORMAP_JET)

    # 生成拼接后的CAM叠加图
    merged_cam_image = show_cam_on_image(rgb_img, merged_heatmap, use_rgb=True)
    merged_cam_image = cv2.cvtColor(merged_cam_image, cv2.COLOR_RGB2BGR)

    # 保存拼接后的结果到图片名目录下
    merged_heatmap_path = os.path.join(
        image_output_dir,
        f'merged_{args.method}_heatmap.jpg'
    )
    merged_cam_path = os.path.join(
        image_output_dir,
        f'merged_{args.method}_cam.jpg'
    )
    original_output_path = os.path.join(
        image_output_dir,
        f'original.jpg'
    )

    print(f"\n{'='*60}")
    print(f"保存所有结果")
    print(f"{'='*60}")
    print(f"\n输出目录: {image_output_dir}")

    print(f"\n【完整图片结果】")
    print(f"  ✓ 完整图片热力图: {os.path.basename(full_heatmap_path)}")
    print(f"  ✓ 完整图片CAM叠加图: {os.path.basename(full_cam_path)}")
    print(f"  ✓ 完整图片原始图像: {os.path.basename(full_original_path)}")

    print(f"\n【4x4切分拼接结果】")
    cv2.imwrite(merged_heatmap_path, merged_heatmap_colored)
    print(f"  ✓ 拼接热力图: {os.path.basename(merged_heatmap_path)}")

    cv2.imwrite(merged_cam_path, merged_cam_image)
    print(f"  ✓ 拼接CAM叠加图: {os.path.basename(merged_cam_path)}")

    cv2.imwrite(
        original_output_path,
        cv2.cvtColor((rgb_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    )
    print(f"  ✓ 原始图像: {os.path.basename(original_output_path)}")

    print(f"\n【4x4切分小图结果】")
    print(f"  ✓ 所有小图结果已保存到: patches/")
    print(f"    共 {len(patches)} 个小图，每个小图包含: original, heatmap, cam")

    print(f"\n{'='*60}")
    print(f"完成！所有可视化结果已保存。")
    print(f"文本描述: {args.image_text}")
    print(f"使用设备: {args.device}")
    print(f"{'='*60}")
