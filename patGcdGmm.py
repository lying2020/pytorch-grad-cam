import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib
# 使用 Agg 后端，防止在无图形界面的服务器上报错，且直接保存文件更稳定
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
from torchvision import transforms
from PIL import Image
import glob
import os
import random

# ==========================================
# 1. 特征提取器 (Feature Extractor)
# ==========================================
class FeatureExtractor:
    def __init__(self, model_name='dino_vitb16'):
        print(f"正在加载模型 {model_name} ...")
        # 加载 DINO 预训练模型 (ViT-B/16)
        self.model = torch.hub.load('facebookresearch/dino:main', model_name)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.features = {}
        self.hook_handles = []

        # [核心逻辑] PartGCD 论文: "fixed patch features that before the last trainable block"
        # ViT-B/16 第 11 层(索引10)的输出是第 12 层的输入
        self.hook_handles.append(
            self.model.blocks[10].register_forward_hook(self._get_features_hook('penultimate_features'))
        )

        # [核心逻辑] PartGCD 论文: "class-to-patch token attention... in the last block"
        # 第 12 层(索引11)的 Attention
        self.hook_handles.append(
            self.model.blocks[11].attn.register_forward_hook(self._get_attn_hook('last_attention'))
        )

    def _get_features_hook(self, name):
        def hook(module, input, output):
            # output: [Batch, N_tokens, Dim]
            self.features[name] = output
        return hook

    def _get_attn_hook(self, name):
        def hook(module, input, output):
            # DINO Attention 返回 (x, attn)
            # attn shape: [Batch, Heads, N, N]
            if isinstance(output, tuple):
                self.features[name] = output[1] # 取索引 1
            else:
                self.features[name] = output
        return hook

    def extract(self, img_tensor):
        """
        输入: img_tensor [B, 3, 224, 224]
        输出: patch_feats [B, 196, 768], cls_attn [B, 196]
        """
        with torch.no_grad():
            _ = self.model(img_tensor)

        # 1. 提取 Patch 特征 (去掉 CLS token)
        # feat shape: [B, 197, 768] -> [B, 196, 768]
        patch_feats = self.features['penultimate_features'][:, 1:, :]

        # 2. 提取 Attention
        attn = self.features['last_attention']

        # 维度检查与处理
        if attn.dim() == 4: # [B, H, N, N]
            attn = attn.mean(dim=1) # Average heads -> [B, N, N]

        # 取 CLS token 对其他 patch 的 attention (row 0, col 1:)
        cls_attn = attn[:, 0, 1:] # [B, 196]

        return patch_feats, cls_attn

# ==========================================
# 2. 图像预处理配置
# ==========================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# ==========================================
# 3. 核心主函数
# ==========================================
def run_partgcd_visualization(image_paths, K=4, output_file="outputs/results/partgcd_result.png", dataset_name=None):
    """
    完整复现 PartGCD Figure 6
    :param image_paths: 图片路径列表
    :param K: 部件数量 (CUB鸟类建议5, 汽车建议6, 其他建议4)
    :param dataset_name: 数据集名称（如 "cars", "birds"），如果为None则从路径中自动提取
    """
    # 如果没有提供数据集名称，尝试从路径中提取
    if dataset_name is None and image_paths:
        # 从第一个路径中提取数据集名称
        first_path = image_paths[0]
        # 尝试从路径中提取数据集名称（例如 "outputs/cars/xxx.jpg" -> "cars"）
        path_parts = first_path.replace('\\', '/').split('/')
        # 查找可能的数据集名称（在outputs、dataset等目录下的子目录）
        for i, part in enumerate(path_parts):
            if part in ['outputs', 'dataset', 'data'] and i + 1 < len(path_parts):
                dataset_name = path_parts[i + 1]
                break
        # 如果还是没找到，使用默认值
        if dataset_name is None:
            dataset_name = "dataset"
    if not image_paths:
        print("错误：没有找到图片，请检查路径。")
        return

    extractor = FeatureExtractor()

    # -------------------------------------------------------
    # 阶段 1: 构建特征池 (Pooling & Filtering)
    # -------------------------------------------------------
    print(f"\n--- 阶段 1: 提取特征 ({len(image_paths)} 张图片) ---")

    pool_features = []
    batch_data = []

    for p in image_paths:
        try:
            img = Image.open(p).convert('RGB')
        except Exception as e:
            print(f"无法读取: {p}, 错误: {e}")
            continue

        # 预处理
        img_t = transform(img).unsqueeze(0)
        if torch.cuda.is_available():
            img_t = img_t.cuda()

        # 提取特征
        feats, attn = extractor.extract(img_t)
        feats = feats.cpu().numpy()[0] # [196, 768]
        attn = attn.cpu().numpy()[0]   # [196]

        # [核心修正 A] 更严格的前景筛选
        # 均值 + 0.1 * 标准差，有效去除背景噪声
        threshold = np.mean(attn) + 0.1 * np.std(attn)
        foreground_mask = attn >= threshold

        # 存下原始数据用于后续画图
        batch_data.append({'img': img, 'feats': feats, 'attn': attn})

        # 只把前景 Patch 放入 GMM 训练池
        if foreground_mask.sum() > 0:
            selected_feats = feats[foreground_mask]
            pool_features.append(selected_feats)
        else:
            # 如果整张图都没有显著前景，保底取TopK
            print(f"警告: 图片 {os.path.basename(p)} 前景激活过低，使用 Top-50")
            idx = np.argsort(attn)[-50:]
            pool_features.append(feats[idx])

    if not pool_features:
        print("未提取到任何有效特征。")
        return

    pool_features = np.concatenate(pool_features, axis=0)
    print(f"特征池构建完成。总 Patch 数: {pool_features.shape[0]}")

    # [核心修正 B] L2 归一化
    # DINO 特征必须归一化，否则 GMM 的欧氏距离计算会因为模长差异产生大量噪声
    print("正在应用 L2 归一化...")
    pool_features = normalize(pool_features, norm='l2')

    # -------------------------------------------------------
    # 阶段 2: 训练 GMM
    # -------------------------------------------------------
    print(f"\n--- 阶段 2: 拟合 GMM (K={K}) ---")
    # 使用 'diag' 协方差，在高维空间更稳定，防止过拟合
    gmm = GaussianMixture(n_components=K, covariance_type='diag', random_state=42, n_init=5, max_iter=200)
    gmm.fit(pool_features)
    print("GMM 训练完成。")

    # -------------------------------------------------------
    # 阶段 3: 识别背景类别
    # -------------------------------------------------------
    print(f"\n--- 阶段 3.1: 识别背景类别 ---")

    # 在所有图片上计算每个类别的统计信息，用于识别背景
    background_scores = np.zeros(K)  # 每个类别的背景得分

    for item in batch_data:
        feats = item['feats']  # [196, 768]
        attn = item['attn']    # [196]

        # L2归一化
        feats = normalize(feats, norm='l2')

        # 预测后验概率
        probs = gmm.predict_proba(feats)  # [196, K]

        # 计算每个类别与低attention区域的相关性
        # 背景类别应该在低attention区域有更高的激活
        for k in range(K):
            part_activations = probs[:, k]  # [196]
            # 计算在低attention区域的平均激活值
            # 使用attention的倒数作为权重（低attention = 高权重）
            attn_inv = 1.0 / (attn + 1e-8)  # 避免除零
            attn_inv_norm = attn_inv / (attn_inv.sum() + 1e-8)
            # 背景得分：在低attention区域的加权平均激活
            background_scores[k] += np.sum(part_activations * attn_inv_norm)

    # 平均化得分
    background_scores = background_scores / len(batch_data)

    # 识别背景类别（得分最高的类别）
    background_class_idx = np.argmax(background_scores)
    print(f"识别到背景类别: Part {background_class_idx + 1} (得分: {background_scores[background_class_idx]:.4f})")
    print(f"所有类别得分: {background_scores}")

    # -------------------------------------------------------
    # 阶段 4: 推理与平滑可视化
    # -------------------------------------------------------
    print(f"\n--- 阶段 4: 生成可视化 ---")
    col_num = 1

    # 定义统一的图片大小
    TARGET_SIZE = (256, 256)

    # 计算总图片数：每张原图有 1个原图 + K个part = K+1 个图
    total_images_per_row = K + 1

    # 生成3组可视化结果，每组随机选择不同的原图
    base_output_file = output_file.replace('.png', '')
    num_groups = 6

    # 为每组随机选择不同的图片索引
    all_indices = list(range(len(batch_data)))
    random.shuffle(all_indices)
    selected_indices_list = []

    # 如果图片数量足够，确保每组选择不同的图片
    if len(batch_data) >= num_groups * col_num:
        # 有足够图片，每组选择不同的图片
        for group_idx in range(num_groups):
            start_idx = group_idx * col_num
            selected_indices = all_indices[start_idx:start_idx + col_num]
            selected_indices_list.append(selected_indices)
    else:
        # 图片数量不足，允许重复但尽量分散
        for group_idx in range(num_groups):
            selected_indices = []
            for i in range(col_num):
                idx = (group_idx * col_num + i) % len(all_indices)
                selected_indices.append(all_indices[idx])
            selected_indices_list.append(selected_indices)

    # -------------------------------------------------------
    # 阶段 3.5: 生成倒数第二层特征的原始heatmap（GMM之前）
    # -------------------------------------------------------
    print(f"\n--- 阶段 3.5: 生成倒数第二层原始特征heatmap (GMM之前) ---")

    # 收集所有用于可视化的图片索引（去重）
    all_selected_indices = set()
    for selected_indices in selected_indices_list[:num_groups]:
        all_selected_indices.update(selected_indices)
    all_selected_indices = sorted(list(all_selected_indices))

    if all_selected_indices:
        print(f"为 {len(all_selected_indices)} 张选中的图片生成原始特征heatmap...")

        # 计算布局（每行显示3张图片）
        num_rows_raw = int(np.ceil(len(all_selected_indices) / 3))
        num_cols_raw = 3

        fig_raw, axes_raw = plt.subplots(num_rows_raw, num_cols_raw,
                                         figsize=(2.5 * num_cols_raw, 2.5 * num_rows_raw),
                                         gridspec_kw={'hspace': 0.1, 'wspace': 0.1,
                                                     'left': 0, 'right': 1, 'top': 0.96, 'bottom': 0})

        # 确保 axes 是 2D 数组
        if num_rows_raw == 1:
            axes_raw = np.expand_dims(axes_raw, axis=0)
        if num_cols_raw == 1:
            axes_raw = np.expand_dims(axes_raw, axis=1)

        axes_raw_flat = axes_raw.flatten()

        title_raw = f'DINO Penultimate Layer Features (Before GMM) - {dataset_name.upper()}'
        # fig_raw.suptitle(title_raw, fontsize=16, fontweight='bold', y=0.99)

        for idx, data_idx in enumerate(all_selected_indices):
            item = batch_data[data_idx]
            original_img = item['img']
            feats = item['feats']  # [196, 768] - 倒数第二层的原始特征

            # 方法1: 计算每个patch特征的L2范数（特征向量的模长）
            # 这能反映每个patch的激活强度
            feat_norms = np.linalg.norm(feats, axis=1)  # [196]

            # Reshape 回 14x14
            h, w = 14, 14
            heatmap_raw = feat_norms.reshape(h, w)

            # 绘制heatmap
            # 高斯模糊
            heatmap_blurred = cv2.GaussianBlur(heatmap_raw, (3, 3), 0)
            # 归一化
            heatmap_norm = (heatmap_blurred - heatmap_blurred.min()) / (heatmap_blurred.max() - heatmap_blurred.min() + 1e-8)
            heatmap_uint8 = np.uint8(255 * heatmap_norm)
            # 上采样
            heatmap_resized = cv2.resize(heatmap_uint8, TARGET_SIZE, interpolation=cv2.INTER_CUBIC)
            # 原图resize
            original_img_resized = original_img.resize(TARGET_SIZE, Image.LANCZOS)

            # 绘制
            axes_raw_flat[idx].imshow(original_img_resized)
            axes_raw_flat[idx].imshow(heatmap_resized, cmap='jet', alpha=0.6)
            axes_raw_flat[idx].axis('off')
            axes_raw_flat[idx].set_aspect('auto')
            axes_raw_flat[idx].margins(0)

        # 隐藏多余的子图
        for idx in range(len(all_selected_indices), len(axes_raw_flat)):
            axes_raw_flat[idx].axis('off')

        plt.subplots_adjust(hspace=0.1, wspace=0.1, left=0, right=1, top=0.92, bottom=0)
        raw_output_file = f"{base_output_file}_penultimate_layer_before_gmm.png"
        plt.savefig(raw_output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"倒数第二层原始特征heatmap已保存至: {raw_output_file}")

        # -------------------------------------------------------
        # 生成重叠的立体效果图（扑克牌堆叠效果，用于论文展示）
        # -------------------------------------------------------
        if len(all_selected_indices) == 6:  # 确保是6张图片
            print(f"生成重叠立体效果图（扑克牌堆叠效果）...")

            # 存储每张图片的heatmap数据
            heatmap_images = []
            original_images = []

            for data_idx in all_selected_indices:
                item = batch_data[data_idx]
                original_img = item['img']
                feats = item['feats']

                # 计算特征L2范数
                feat_norms = np.linalg.norm(feats, axis=1)
                h, w = 14, 14
                heatmap_raw = feat_norms.reshape(h, w)

                # 高斯模糊
                heatmap_blurred = cv2.GaussianBlur(heatmap_raw, (3, 3), 0)
                # 归一化
                heatmap_norm = (heatmap_blurred - heatmap_blurred.min()) / (heatmap_blurred.max() - heatmap_blurred.min() + 1e-8)
                heatmap_uint8 = np.uint8(255 * heatmap_norm)
                # 上采样到合适的尺寸
                large_size = (600, 600)
                heatmap_resized = cv2.resize(heatmap_uint8, large_size, interpolation=cv2.INTER_CUBIC)

                # 转换为RGB colormap
                heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
                heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

                # 原图也resize
                original_img_resized = original_img.resize(large_size, Image.LANCZOS)
                original_img_array = np.array(original_img_resized)

                heatmap_images.append(heatmap_colored)
                original_images.append(original_img_array)

            # 95%重叠意味着只有5%的偏移，每张图片向右下偏移约5%的宽度
            offset_per_layer = int(large_size[0] * 0.05)  # 约30像素
            max_offset = offset_per_layer * 6  # 6张图片的最大累积偏移
            # 画布大小：图片大小 + 最大偏移 + 边距
            canvas_size = (large_size[0] + max_offset + 100, large_size[1] + max_offset + 100)
            canvas = np.ones((canvas_size[1], canvas_size[0], 3), dtype=np.float32) * 255.0

            # 计算中心位置（稍微偏左上方，为向右下方的偏移留出空间）
            center_x = canvas_size[0] // 2 - max_offset // 2
            center_y = canvas_size[1] // 2 - max_offset // 2

            # 扑克牌堆叠效果：每张图片向右下方轻微偏移（水平方向向内推）
            # 95%重叠，每层偏移约5%的宽度
            base_offset_x = offset_per_layer
            base_offset_y = offset_per_layer // 2  # 垂直偏移稍小，形成水平推动的效果

            # 透明度设置：后面的图片稍微透明一些，形成层次感
            alphas = [0.9, 0.85, 0.8, 0.75, 0.7, 0.65]  # 从前往后逐渐降低

            # 透视效果：后面的图片稍微小一点
            scales = [1.0, 0.98, 0.96, 0.94, 0.92, 0.90]  # 从前往后逐渐缩小

            for idx, (heatmap_img, original_img_array) in enumerate(zip(heatmap_images, original_images)):
                # 只使用heatmap，不混合原图
                blended = heatmap_img.copy()

                # 缩放（透视效果）
                scale = scales[idx]
                if scale < 1.0:
                    scaled_size = (int(large_size[0] * scale), int(large_size[1] * scale))
                    blended = cv2.resize(blended, scaled_size, interpolation=cv2.INTER_LINEAR)
                else:
                    scaled_size = large_size

                # 计算在画布上的位置（向右下方偏移，形成堆叠效果）
                offset_x = idx * base_offset_x
                offset_y = idx * base_offset_y
                x_start = center_x - scaled_size[0]//2 + offset_x
                y_start = center_y - scaled_size[1]//2 + offset_y
                x_end = x_start + scaled_size[0]
                y_end = y_start + scaled_size[1]

                # 确保不越界
                x_start_clip = max(0, x_start)
                y_start_clip = max(0, y_start)
                x_end_clip = min(canvas_size[0], x_end)
                y_end_clip = min(canvas_size[1], y_end)

                # 计算需要从图片中裁剪的区域
                crop_x_start = max(0, -x_start)
                crop_y_start = max(0, -y_start)
                crop_x_end = crop_x_start + (x_end_clip - x_start_clip)
                crop_y_end = crop_y_start + (y_end_clip - y_start_clip)

                # 叠加到画布上
                alpha = alphas[idx]
                canvas_region = canvas[y_start_clip:y_end_clip, x_start_clip:x_end_clip]
                blended_region = blended[crop_y_start:crop_y_end, crop_x_start:crop_x_end].astype(np.float32)

                # Alpha混合
                canvas[y_start_clip:y_end_clip, x_start_clip:x_end_clip] = (
                    canvas_region * (1 - alpha) + blended_region * alpha
                )

            # 转换为uint8
            canvas = np.clip(canvas, 0, 255).astype(np.uint8)

            # 保存结果
            fig_overlap, ax_overlap = plt.subplots(1, 1, figsize=(10, 10))
            ax_overlap.imshow(canvas)
            ax_overlap.axis('off')
            ax_overlap.set_aspect('equal')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

            overlap_output_file = f"{base_output_file}_penultimate_layer_overlap.png"
            plt.savefig(overlap_output_file, dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()
            print(f"重叠立体效果图（扑克牌堆叠）已保存至: {overlap_output_file}")

            # -------------------------------------------------------
            # 生成原图的堆叠图（使用相同的堆叠方式）
            # -------------------------------------------------------
            print(f"生成原图堆叠图（扑克牌堆叠效果）...")

            # 使用相同的堆叠参数
            canvas_orig = np.ones((canvas_size[1], canvas_size[0], 3), dtype=np.float32) * 255.0

            for idx, original_img_array in enumerate(original_images):
                # 缩放（透视效果）
                scale = scales[idx]
                if scale < 1.0:
                    scaled_size = (int(large_size[0] * scale), int(large_size[1] * scale))
                    scaled_img = cv2.resize(original_img_array, scaled_size, interpolation=cv2.INTER_LINEAR)
                else:
                    scaled_size = large_size
                    scaled_img = original_img_array.copy()

                # 计算在画布上的位置（向右下方偏移，形成堆叠效果）
                offset_x = idx * base_offset_x
                offset_y = idx * base_offset_y
                x_start = center_x - scaled_size[0]//2 + offset_x
                y_start = center_y - scaled_size[1]//2 + offset_y
                x_end = x_start + scaled_size[0]
                y_end = y_start + scaled_size[1]

                # 确保不越界
                x_start_clip = max(0, x_start)
                y_start_clip = max(0, y_start)
                x_end_clip = min(canvas_size[0], x_end)
                y_end_clip = min(canvas_size[1], y_end)

                # 计算需要从图片中裁剪的区域
                crop_x_start = max(0, -x_start)
                crop_y_start = max(0, -y_start)
                crop_x_end = crop_x_start + (x_end_clip - x_start_clip)
                crop_y_end = crop_y_start + (y_end_clip - y_start_clip)

                # 叠加到画布上
                alpha = alphas[idx]
                canvas_region = canvas_orig[y_start_clip:y_end_clip, x_start_clip:x_end_clip]
                scaled_region = scaled_img[crop_y_start:crop_y_end, crop_x_start:crop_x_end].astype(np.float32)

                # Alpha混合
                canvas_orig[y_start_clip:y_end_clip, x_start_clip:x_end_clip] = (
                    canvas_region * (1 - alpha) + scaled_region * alpha
                )

            # 转换为uint8
            canvas_orig = np.clip(canvas_orig, 0, 255).astype(np.uint8)

            # 保存结果
            fig_orig_overlap, ax_orig_overlap = plt.subplots(1, 1, figsize=(10, 10))
            ax_orig_overlap.imshow(canvas_orig)
            ax_orig_overlap.axis('off')
            ax_orig_overlap.set_aspect('equal')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

            orig_overlap_output_file = f"{base_output_file}_original_images_overlap.png"
            plt.savefig(orig_overlap_output_file, dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()
            print(f"原图堆叠图（扑克牌堆叠）已保存至: {orig_overlap_output_file}")
        else:
            print(f"注意：当前选中了 {len(all_selected_indices)} 张图片，不是6张，跳过旋转重叠图的生成")

    # 辅助函数：绘制单个part的热力图
    def draw_heatmap(ax, original_img, heatmap):
        """在ax上绘制原图和热力图的叠加"""
        # 步骤 1: 高斯模糊
        heatmap = cv2.GaussianBlur(heatmap, (3, 3), 0)
        # 步骤 2: 归一化到 0-255
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap_uint8 = np.uint8(255 * heatmap_norm)
        # 步骤 3: 双立方插值上采样到统一大小
        heatmap_resized = cv2.resize(heatmap_uint8, TARGET_SIZE, interpolation=cv2.INTER_CUBIC)
        # 原图也resize到统一大小
        original_img_resized = original_img.resize(TARGET_SIZE, Image.LANCZOS)
        # 绘制
        ax.imshow(original_img_resized)
        ax.imshow(heatmap_resized, cmap='jet', alpha=0.6)
        ax.axis('off')
        ax.set_aspect('auto')
        ax.margins(0)

    for group_idx, selected_indices in enumerate(selected_indices_list[:num_groups]):
        display_count = len(selected_indices)

        # 存储该组的数据，用于后续生成单独的组图
        group_data = []
        # 存储背景类别的数据
        background_data = []

        # 计算总图片数
        total_images = display_count * total_images_per_row

        # 计算布局
        num_rows = col_num
        num_cols = int(np.ceil(total_images / num_rows))

        # 创建紧凑的布局（无间距）
        # top=0.92 为标题留出空间，避免重叠
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(2.5 * num_cols, 2.5 * num_rows),
                                 gridspec_kw={'hspace': 0, 'wspace': 0, 'left': 0, 'right': 1, 'top': 0.96, 'bottom': 0})

        # 添加总标题（包含数据集名称）
        # y=0.99 将标题放在更上方，避免与图片重叠
        title = f'PartGCD Visualization - {dataset_name.upper()} (Group {group_idx+1})'
        # fig.suptitle(title, fontsize=16, fontweight='bold', y=0.99)

        # 确保 axes 是 2D 数组
        if num_rows == 1:
            axes = np.expand_dims(axes, axis=0)
        if num_cols == 1:
            axes = np.expand_dims(axes, axis=1)

        # 扁平化axes以便索引
        axes_flat = axes.flatten()

        # 当前图片索引
        img_idx = 0

        for i, data_idx in enumerate(selected_indices):
            item = batch_data[data_idx]
            original_img = item['img']
            feats = item['feats'] # [196, 768]

            # [重要] 推理时也必须做 L2 归一化，与训练保持一致
            feats = normalize(feats, norm='l2')

            # 预测后验概率 P(Part_k | Patch)
            probs = gmm.predict_proba(feats) # [196, K]

            # Reshape 回 14x14
            h, w = 14, 14
            part_maps = probs.reshape(h, w, K)

            # 保存数据用于后续生成单独的组图
            group_data.append({
                'original_img': original_img,
                'part_maps': part_maps
            })

            # 保存背景类别的数据
            background_data.append({
                'original_img': original_img,
                'background_map': part_maps[:, :, background_class_idx]
            })

            # 1. 绘制原图（统一resize到TARGET_SIZE）
            original_img_resized = original_img.resize(TARGET_SIZE, Image.LANCZOS)
            ax_orig = axes_flat[img_idx]
            ax_orig.imshow(original_img_resized)
            ax_orig.axis('off')
            ax_orig.set_aspect('auto')  # 确保图片填满子图
            ax_orig.margins(0)  # 移除边距
            img_idx += 1

            # 2. 绘制各个 Part（显示所有K个part）
            for k in range(K):
                ax_part = axes_flat[img_idx]
                heatmap = part_maps[:, :, k]
                draw_heatmap(ax_part, original_img, heatmap)
                img_idx += 1

        # 隐藏多余的子图
        for idx in range(img_idx, len(axes_flat)):
            axes_flat[idx].axis('off')

        # 完全无间距布局，top=0.92 为标题留出空间
        plt.subplots_adjust(hspace=0, wspace=0, left=0, right=1, top=0.92, bottom=0)  # 完全无间距

        # 生成不同的输出文件名
        group_output_file = f"{base_output_file}_group{group_idx+1}.png"
        plt.savefig(group_output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"可视化结果已保存至: {group_output_file} (包含 {display_count} 张原图)")

        # 生成背景类别的单独可视化
        if background_data:
            fig_bg, axes_bg = plt.subplots(display_count, 1, figsize=(2.5, 2.5 * display_count),
                                          gridspec_kw={'hspace': 0.1, 'wspace': 0, 'left': 0, 'right': 1, 'top': 0.96, 'bottom': 0})
            if display_count == 1:
                axes_bg = [axes_bg]
            title_bg = f'PartGCD Background - {dataset_name.upper()} (Group {group_idx+1}, Part {background_class_idx+1})'
            # fig_bg.suptitle(title_bg, fontsize=16, fontweight='bold', y=0.99)
            for i, data in enumerate(background_data):
                heatmap = 1.0 - data['background_map']
                draw_heatmap(axes_bg[i], data['original_img'], heatmap)
            plt.subplots_adjust(hspace=0.1, wspace=0, left=0, right=1, top=0.92, bottom=0)
            bg_output_file = f"{base_output_file}_group{group_idx+1}_background.png"
            plt.savefig(bg_output_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"背景类别可视化已保存至: {bg_output_file}")

        # 生成非背景part的堆叠图（用于论文展示）
        # 对每张图片单独生成其非背景part的堆叠图
        if group_data:
            print(f"生成非背景part堆叠图（Group {group_idx+1}）...")

            # 对每张图片单独处理
            for img_idx, data in enumerate(group_data):
                part_maps = data['part_maps']  # [h, w, K]

                # 收集这张图片的所有非背景part热力图
                non_background_heatmaps = []
                for k in range(K):
                    if k != background_class_idx:
                        heatmap = part_maps[:, :, k]  # [h, w]
                        non_background_heatmaps.append(heatmap)

                # 如果非背景part数量合理，生成堆叠图
                if len(non_background_heatmaps) > 0 and len(non_background_heatmaps) <= 20:
                    # 将热力图转换为RGB图像
                    large_size = (600, 600)
                    heatmap_images = []

                    for heatmap in non_background_heatmaps:
                        # 高斯模糊
                        heatmap_blurred = cv2.GaussianBlur(heatmap, (3, 3), 0)
                        # 归一化
                        heatmap_norm = (heatmap_blurred - heatmap_blurred.min()) / (heatmap_blurred.max() - heatmap_blurred.min() + 1e-8)
                        heatmap_uint8 = np.uint8(255 * heatmap_norm)
                        # 上采样
                        heatmap_resized = cv2.resize(heatmap_uint8, large_size, interpolation=cv2.INTER_CUBIC)
                        # 转换为RGB colormap
                        heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
                        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                        heatmap_images.append(heatmap_colored)

                    # 计算堆叠参数
                    num_heatmaps = len(heatmap_images)
                    offset_per_layer = int(large_size[0] * 0.05)  # 约30像素
                    max_offset = offset_per_layer * num_heatmaps
                    canvas_size = (large_size[0] + max_offset + 100, large_size[1] + max_offset + 100)
                    canvas = np.ones((canvas_size[1], canvas_size[0], 3), dtype=np.float32) * 255.0

                    # 计算中心位置
                    center_x = canvas_size[0] // 2 - max_offset // 2
                    center_y = canvas_size[1] // 2 - max_offset // 2

                    # 堆叠参数
                    base_offset_x = offset_per_layer
                    base_offset_y = offset_per_layer // 2

                    # 透明度和缩放设置
                    alphas = [0.9 - i * 0.05 for i in range(num_heatmaps)]
                    alphas = [max(0.3, a) for a in alphas]  # 最小透明度0.3
                    scales = [1.0 - i * 0.02 for i in range(num_heatmaps)]
                    scales = [max(0.8, s) for s in scales]  # 最小缩放0.8

                    # 堆叠所有非背景part
                    for idx, heatmap_img in enumerate(heatmap_images):
                        # 缩放
                        scale = scales[idx]
                        if scale < 1.0:
                            scaled_size = (int(large_size[0] * scale), int(large_size[1] * scale))
                            blended = cv2.resize(heatmap_img, scaled_size, interpolation=cv2.INTER_LINEAR)
                        else:
                            scaled_size = large_size
                            blended = heatmap_img.copy()

                        # 计算位置
                        offset_x = idx * base_offset_x
                        offset_y = idx * base_offset_y
                        x_start = center_x - scaled_size[0]//2 + offset_x
                        y_start = center_y - scaled_size[1]//2 + offset_y
                        x_end = x_start + scaled_size[0]
                        y_end = y_start + scaled_size[1]

                        # 确保不越界
                        x_start_clip = max(0, x_start)
                        y_start_clip = max(0, y_start)
                        x_end_clip = min(canvas_size[0], x_end)
                        y_end_clip = min(canvas_size[1], y_end)

                        # 计算裁剪区域
                        crop_x_start = max(0, -x_start)
                        crop_y_start = max(0, -y_start)
                        crop_x_end = crop_x_start + (x_end_clip - x_start_clip)
                        crop_y_end = crop_y_start + (y_end_clip - y_start_clip)

                        # 叠加到画布上
                        alpha = alphas[idx]
                        canvas_region = canvas[y_start_clip:y_end_clip, x_start_clip:x_end_clip]
                        blended_region = blended[crop_y_start:crop_y_end, crop_x_start:crop_x_end].astype(np.float32)

                        # Alpha混合
                        canvas[y_start_clip:y_end_clip, x_start_clip:x_end_clip] = (
                            canvas_region * (1 - alpha) + blended_region * alpha
                        )

                    # 转换为uint8
                    canvas = np.clip(canvas, 0, 255).astype(np.uint8)

                    # 保存结果
                    fig_parts, ax_parts = plt.subplots(1, 1, figsize=(10, 10))
                    ax_parts.imshow(canvas)
                    ax_parts.axis('off')
                    ax_parts.set_aspect('equal')
                    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

                    parts_output_file = f"{base_output_file}_group{group_idx+1}_img{img_idx+1}_non_background_parts_overlap.png"
                    plt.savefig(parts_output_file, dpi=300, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    print(f"非背景part堆叠图已保存至: {parts_output_file} (包含 {num_heatmaps} 个part)")
                else:
                    print(f"注意：图片 {img_idx+1} 的非背景part数量为 {len(non_background_heatmaps)}，跳过堆叠图生成（数量过多或为0）")

        # 生成单独的组图：original 和 每个 part
        # 1. Original组图（3行1列，只显示原图）
        # fig_orig, axes_orig = plt.subplots(display_count, 1, figsize=(2.5, 2.5 * display_count),
        #                                    gridspec_kw={'hspace': 0, 'wspace': 0, 'left': 0, 'right': 1, 'top': 0.96, 'bottom': 0})
        # if display_count == 1:
        #     axes_orig = [axes_orig]
        # title_orig = f'PartGCD Visualization - {dataset_name.upper()} (Group {group_idx+1} - Original)'
        # fig_orig.suptitle(title_orig, fontsize=16, fontweight='bold', y=0.99)
        # for i, data in enumerate(group_data):
        #     original_img_resized = data['original_img'].resize(TARGET_SIZE, Image.LANCZOS)
        #     axes_orig[i].imshow(original_img_resized)
        #     axes_orig[i].axis('off')
        #     axes_orig[i].set_aspect('auto')
        #     axes_orig[i].margins(0)
        # plt.subplots_adjust(hspace=0, wspace=0, left=0, right=1, top=0.92, bottom=0)
        # orig_output_file = f"{base_output_file}_group{group_idx+1}_original.png"
        # plt.savefig(orig_output_file, dpi=300, bbox_inches='tight')
        # plt.close()
        # print(f"可视化结果已保存至: {orig_output_file}")

        # # 2. 每个Part的组图（3行1列，只显示该part）
        # for k in range(K):
        #     fig_part, axes_part = plt.subplots(display_count, 1, figsize=(2.5, 2.5 * display_count),
        #                                        gridspec_kw={'hspace': 0, 'wspace': 0, 'left': 0, 'right': 1, 'top': 0.96, 'bottom': 0})
        #     if display_count == 1:
        #         axes_part = [axes_part]
        #     title_part = f'PartGCD Visualization - {dataset_name.upper()} (Group {group_idx+1} - Part {k+1})'
        #     fig_part.suptitle(title_part, fontsize=16, fontweight='bold', y=0.99)
        #     for i, data in enumerate(group_data):
        #         heatmap = data['part_maps'][:, :, k]
        #         draw_heatmap(axes_part[i], data['original_img'], heatmap)
        #     plt.subplots_adjust(hspace=0, wspace=0, left=0, right=1, top=0.92, bottom=0)
        #     part_output_file = f"{base_output_file}_group{group_idx+1}_part{k+1}.png"
            #     plt.savefig(part_output_file, dpi=300, bbox_inches='tight')
            #     plt.close()
            #     print(f"可视化结果已保存至: {part_output_file}")

    # -------------------------------------------------------
    # 阶段 5: 生成全局背景类别可视化（所有图片）
    # -------------------------------------------------------
    print(f"\n--- 阶段 5: 生成全局背景类别可视化 ---")

    # 为所有图片生成背景可视化
    all_background_data = []
    for item in batch_data:
        original_img = item['img']
        feats = item['feats']

        # L2归一化
        feats = normalize(feats, norm='l2')

        # 预测后验概率
        probs = gmm.predict_proba(feats)  # [196, K]

        # Reshape 回 14x14
        h, w = 14, 14
        part_maps = probs.reshape(h, w, K)

        all_background_data.append({
            'original_img': original_img,
            'background_map': part_maps[:, :, background_class_idx]
        })

    # # 生成全局背景可视化（最多显示20张图片，避免图片过多）
    # max_display = min(20, len(all_background_data))
    # selected_background_data = all_background_data[:max_display]

    # if selected_background_data:
    #     num_rows_bg = int(np.ceil(np.sqrt(max_display)))
    #     num_cols_bg = int(np.ceil(max_display / num_rows_bg))

    #     fig_bg_all, axes_bg_all = plt.subplots(num_rows_bg, num_cols_bg,
    #                                            figsize=(2.5 * num_cols_bg, 2.5 * num_rows_bg),
    #                                            gridspec_kw={'hspace': 0.1, 'wspace': 0.1,
    #                                                        'left': 0, 'right': 1, 'top': 0.96, 'bottom': 0})

    #     # 确保 axes 是 2D 数组
    #     if num_rows_bg == 1:
    #         axes_bg_all = np.expand_dims(axes_bg_all, axis=0)
    #     if num_cols_bg == 1:
    #         axes_bg_all = np.expand_dims(axes_bg_all, axis=1)

    #     axes_bg_flat = axes_bg_all.flatten()

    #     title_bg_all = f'PartGCD Background - {dataset_name.upper()} (All Images, Part {background_class_idx+1})'
    #     # fig_bg_all.suptitle(title_bg_all, fontsize=16, fontweight='bold', y=0.99)

    #     for i, data in enumerate(selected_background_data):
    #         heatmap = data['background_map']
    #         draw_heatmap(axes_bg_flat[i], data['original_img'], heatmap)

    #     # 隐藏多余的子图
    #     for idx in range(len(selected_background_data), len(axes_bg_flat)):
    #         axes_bg_flat[idx].axis('off')

    #     plt.subplots_adjust(hspace=0.1, wspace=0.1, left=0, right=1, top=0.92, bottom=0)
    #     bg_all_output_file = f"{base_output_file}_background_all.png"
    #     plt.savefig(bg_all_output_file, dpi=300, bbox_inches='tight')
    #     plt.close()
    #     print(f"全局背景类别可视化已保存至: {bg_all_output_file} (显示 {max_display} 张图片)")

# ==========================================
# 4. 运行入口
# ==========================================
if __name__ == "__main__":
    # -------------------------------------------------
    # 配置: 修改这里的路径匹配模式
    # -------------------------------------------------
    # 例如: "dataset/birds/*.jpg" 或者简单的 "*.jpg"
    img_path_pattern = "outputs/stanford_cars/*.jpg"
    # img_path_pattern = "outputs/fgvc_aircraft/*.jpg"
    # img_path_pattern = "outputs/oxford_pets_cat/*.jpg"
    # img_path_pattern = "outputs/oxford_pets_dog/*.jpg"
    # img_path_patterns = ["outputs/stanford_cars/*.jpg", "outputs/fgvc_aircraft/*.jpg", "outputs/oxford_pets/*.jpg", "outputs/oxford_pets_cat/*.jpg", "outputs/oxford_pets_dog/*.jpg"]
    img_path_patterns = ["outputs/fgvc_aircraft/*.jpg"]

    for img_path_pattern in img_path_patterns:
        # 搜索图片
        img_list = glob.glob(img_path_pattern)

        if len(img_list) > 0:
            print(f"找到 {len(img_list)} 张图片: {img_list[:3]} ...")

            # 从路径中提取数据集名称
            dataset_name = None
            if 'stanford_cars' in img_path_pattern.lower():
                dataset_name = "stanford_cars"
            elif 'birds' in img_path_pattern.lower() or 'cub' in img_path_pattern.lower():
                dataset_name = "birds"
            elif 'oxford_pets_cat' in img_path_pattern.lower():
                dataset_name = "oxford_pets_cat"
            elif 'oxford_pets_dog' in img_path_pattern.lower():
                dataset_name = "oxford_pets_dog"
            elif 'fgvc_aircraft' in img_path_pattern.lower():
                dataset_name = "fgvc_aircraft"
            elif 'dataset' in img_path_pattern:
                # 尝试从路径中提取
                path_parts = img_path_pattern.replace('\\', '/').split('/')
                for i, part in enumerate(path_parts):
                    if part in ['outputs', 'dataset', 'data'] and i + 1 < len(path_parts):
                        dataset_name = path_parts[i + 1]
                        break

            # CUB鸟类建议 K=5
            # Stanford Cars 建议 K=6
            # 其他通用物体 建议 K=4
            run_partgcd_visualization(img_list, K=5, output_file=f"outputs/results/{dataset_name}_partgcd_reproduction.png", dataset_name=dataset_name)
        else:
            print(f"当前目录下未找到匹配 '{img_path_pattern}' 的图片，请修改代码底部的 img_path_pattern 变量。")