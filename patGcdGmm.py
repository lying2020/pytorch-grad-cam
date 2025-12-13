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
def run_partgcd_visualization(image_paths, K=4, output_file="partgcd_result.png"):
    """
    完整复现 PartGCD Figure 6
    :param image_paths: 图片路径列表
    :param K: 部件数量 (CUB鸟类建议5, 汽车建议6, 其他建议4)
    """
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
    # 阶段 3: 推理与平滑可视化
    # -------------------------------------------------------
    print(f"\n--- 阶段 3: 生成可视化 ---")

    # 随机选 3 张图展示（或全部展示）
    display_count = min(3, len(batch_data))
    fig, axes = plt.subplots(display_count, K + 1, figsize=(3 * (K+1), 3 * display_count))

    # 添加总标题
    fig.suptitle('PartGCD Visualization', fontsize=16, fontweight='bold', y=0.98)

    # 确保 axes 是 2D 数组
    if display_count == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(display_count):
        item = batch_data[i]
        original_img = item['img']
        feats = item['feats'] # [196, 768]

        # [重要] 推理时也必须做 L2 归一化，与训练保持一致
        feats = normalize(feats, norm='l2')

        # 预测后验概率 P(Part_k | Patch)
        probs = gmm.predict_proba(feats) # [196, K]

        # Reshape 回 14x14
        h, w = 14, 14
        part_maps = probs.reshape(h, w, K)

        # 1. 绘制原图
        ax_orig = axes[i, 0]
        ax_orig.imshow(original_img)
        ax_orig.axis('off')

        # 2. 绘制各个 Part
        for k in range(K):
            ax_part = axes[i, k+1]

            # 取出第 k 个 part 的热力图
            heatmap = part_maps[:, :, k]

            # [核心修正 C] 可视化后处理三部曲

            # 步骤 1: 高斯模糊 (消除 Patch 边缘，让热力图呈云团状)
            heatmap = cv2.GaussianBlur(heatmap, (3, 3), 0)

            # 步骤 2: 归一化到 0-255
            heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            heatmap_uint8 = np.uint8(255 * heatmap_norm)

            # 步骤 3: 双立方插值上采样 (比 Linear 更平滑)
            heatmap_resized = cv2.resize(heatmap_uint8, (original_img.size[0], original_img.size[1]), interpolation=cv2.INTER_CUBIC)

            # 绘制 (使用 matplotlib 的 alpha 混合，效果最干净)
            ax_part.imshow(original_img)
            # jet 颜色映射，alpha=0.6 实现半透明叠加
            im = ax_part.imshow(heatmap_resized, cmap='jet', alpha=0.6)

            ax_part.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为总标题留出空间
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"可视化结果已保存至: {output_file}")

# ==========================================
# 4. 运行入口
# ==========================================
if __name__ == "__main__":
    # -------------------------------------------------
    # 配置: 修改这里的路径匹配模式
    # -------------------------------------------------
    # 例如: "dataset/birds/*.jpg" 或者简单的 "*.jpg"
    img_path_pattern = "cars/*.jpg"

    # 搜索图片
    img_list = glob.glob(img_path_pattern)

    if len(img_list) > 0:
        print(f"找到 {len(img_list)} 张图片: {img_list[:3]} ...")

        # CUB鸟类建议 K=5
        # Stanford Cars 建议 K=6
        # 其他通用物体 建议 K=4
        run_partgcd_visualization(img_list, K=5, output_file="outputs/partgcd_reproduction.png")
    else:
        print(f"当前目录下未找到匹配 '{img_path_pattern}' 的图片，请修改代码底部的 img_path_pattern 变量。")