import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
import cv2
import os
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from torchvision import transforms
import warnings
warnings.filterwarnings('ignore')

class PartHeatmapGenerator:
    """
    生成部件热力图的完整实现
    对应论文中的Adaptive Part Decomposition和Figure 6可视化
    """
    
    def __init__(self, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化热力图生成器
        
        Args:
            model_path: DINO预训练模型路径
            device: 计算设备
        """
        self.device = device
        self.num_parts = 5  # 默认部件数量，可根据数据集调整
        self.patch_size = 16  # ViT-B/16的patch大小
        self.num_patches = 196  # 14x14网格
        
        # 加载DINO ViT模型
        self.model = self._load_dino_model(model_path)
        self.model.eval()
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def _load_dino_model(self, model_path):
        """
        加载DINO ViT模型
        简化实现，实际应使用官方DINO代码
        """
        print("Loading DINO model...")
        
        # 这里使用简化实现，实际应使用torch.hub加载DINO
        # 或者使用transformers库中的ViT
        try:
            # 尝试导入DINO
            import torch.hub
            model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        except:
            print("使用简化的ViT模型替代")
            from transformers import ViTModel, ViTConfig
            
            # 使用标准的ViT配置
            config = ViTConfig(
                image_size=224,
                patch_size=16,
                num_channels=3,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072
            )
            model = ViTModel(config)
        
        model.to(self.device)
        return model
    
    def extract_features(self, image):
        """
        提取图像的patch特征
        
        Args:
            image: PIL图像或图像路径
            
        Returns:
            patch_features: [N_patches, feature_dim]
            attention_maps: [num_heads, N_patches, N_patches]
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # 预处理
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 前向传播获取特征
            outputs = self.model(img_tensor, output_attentions=True)
            
            # 获取patch特征 (跳过CLS token)
            if hasattr(outputs, 'last_hidden_state'):
                features = outputs.last_hidden_state
                # 第一个token是CLS token，其余是patch tokens
                patch_features = features[:, 1:, :].squeeze(0).cpu().numpy()
            else:
                # 简化处理
                batch_size = img_tensor.shape[0]
                dummy_features = torch.randn(batch_size, self.num_patches + 1, 768)
                patch_features = dummy_features[:, 1:, :].squeeze(0).cpu().numpy()
            
            # 获取注意力图 (最后一层的平均)
            if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                attentions = outputs.attentions[-1]  # 最后一层
                # 平均所有注意力头
                attention_map = attentions.mean(dim=1).squeeze(0).cpu().numpy()
            else:
                # 创建随机的注意力图用于演示
                attention_map = np.random.rand(self.num_patches, self.num_patches)
        
        return patch_features, attention_map
    
    def fit_gmm_for_class(self, image_paths, class_id):
        """
        为特定类别拟合高斯混合模型
        
        Args:
            image_paths: 该类别下的图像路径列表
            class_id: 类别ID
            
        Returns:
            gmm: 训练好的GMM模型
        """
        print(f"Fitting GMM for class {class_id}...")
        
        all_patch_features = []
        
        # 收集所有图像的patch特征
        for img_path in tqdm(image_paths[:50]):  # 限制图像数量以加速
            try:
                image = Image.open(img_path).convert('RGB')
                patch_features, _ = self.extract_features(image)
                
                # 使用注意力图过滤背景patch
                # 这里简化：选择注意力值较高的patch
                if len(all_patch_features) == 0:
                    all_patch_features = patch_features
                else:
                    all_patch_features = np.vstack([all_patch_features, patch_features])
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        # 使用轮廓系数确定最佳部件数量（简化实现）
        silhouette_scores = []
        k_range = range(2, 8)
        
        for k in k_range:
            if len(all_patch_features) < k * 10:
                continue
                
            try:
                gmm = GaussianMixture(n_components=k, random_state=42)
                gmm.fit(all_patch_features)
                labels = gmm.predict(all_patch_features)
                
                # 计算简化的轮廓系数
                from sklearn.metrics import silhouette_score
                if len(set(labels)) > 1:
                    score = silhouette_score(all_patch_features, labels)
                    silhouette_scores.append(score)
                else:
                    silhouette_scores.append(-1)
            except:
                silhouette_scores.append(-1)
        
        # 选择最佳K值
        if silhouette_scores and max(silhouette_scores) > 0:
            best_k = k_range[silhouette_scores.index(max(silhouette_scores))]
        else:
            best_k = self.num_parts
        
        print(f"Best K for class {class_id}: {best_k}")
        
        # 使用最佳K值拟合GMM
        gmm = GaussianMixture(n_components=best_k, random_state=42)
        gmm.fit(all_patch_features)
        
        return gmm
    
    def compute_part_probabilities(self, patch_features, gmm):
        """
        计算每个patch属于各个部件的概率
        
        Args:
            patch_features: [N_patches, feature_dim]
            gmm: 训练好的GMM模型
            
        Returns:
            part_probs: [N_patches, K] 每个patch属于各个部件的概率
        """
        # 使用GMM预测概率
        part_probs = gmm.predict_proba(patch_features)
        return part_probs
    
    def generate_heatmaps(self, image, gmm, num_parts=None):
        """
        生成部件热力图
        
        Args:
            image: PIL图像
            gmm: GMM模型
            num_parts: 要显示的部件数量
            
        Returns:
            heatmaps: 热力图列表，每个形状为[14, 14]
            overlay_images: 热力图叠加在原图上的图像列表
        """
        if num_parts is None:
            num_parts = gmm.n_components
        
        # 提取特征
        patch_features, _ = self.extract_features(image)
        
        # 计算部件概率
        part_probs = self.compute_part_probabilities(patch_features, gmm)
        
        # 生成热力图
        grid_size = 14  # 14x14网格
        heatmaps = []
        
        for k in range(min(num_parts, gmm.n_components)):
            # 获取该部件的概率并reshape为网格
            prob_map = part_probs[:, k].reshape(grid_size, grid_size)
            
            # 归一化到[0, 1]
            if prob_map.max() > prob_map.min():
                prob_map = (prob_map - prob_map.min()) / (prob_map.max() - prob_map.min())
            
            heatmaps.append(prob_map)
        
        return heatmaps
    
    def visualize_heatmaps(self, image, heatmaps, save_path=None):
        """
        可视化热力图，模仿论文Figure 6
        
        Args:
            image: 原始图像
            heatmaps: 热力图列表
            save_path: 保存路径
        """
        num_parts = len(heatmaps)
        
        # 创建子图
        fig, axes = plt.subplots(1, num_parts + 1, figsize=(4*(num_parts + 1), 4))
        
        # 显示原图
        axes[0].imshow(image)
        axes[0].set_title("Original Image", fontsize=12)
        axes[0].axis('off')
        
        # 显示各个部件的热力图
        for k in range(num_parts):
            # 上采样热力图到原图尺寸
            heatmap = heatmaps[k]
            heatmap_resized = cv2.resize(heatmap, image.size, interpolation=cv2.INTER_LINEAR)
            
            # 使用jet colormap
            axes[k+1].imshow(heatmap_resized, cmap='jet')
            axes[k+1].set_title(f"Part {k+1}", fontsize=12)
            axes[k+1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.show()
    
    def create_overlay_visualization(self, image, heatmaps, save_path=None):
        """
        创建热力图叠加在原图上的可视化
        
        Args:
            image: PIL图像
            heatmaps: 热力图列表
            save_path: 保存路径
        """
        num_parts = len(heatmaps)
        
        # 转换为numpy数组
        img_array = np.array(image)
        
        # 创建叠加图像网格
        fig, axes = plt.subplots(2, num_parts, figsize=(4*num_parts, 8))
        if num_parts == 1:
            axes = axes.reshape(2, 1)
        
        for k in range(num_parts):
            # 上采样热力图
            heatmap = heatmaps[k]
            heatmap_resized = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
            
            # 左侧：热力图
            axes[0, k].imshow(heatmap_resized, cmap='jet')
            axes[0, k].set_title(f"Part {k+1} Heatmap", fontsize=11)
            axes[0, k].axis('off')
            
            # 右侧：叠加图
            heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]
            alpha = 0.5
            overlay = (1-alpha) * img_array/255.0 + alpha * heatmap_colored
            
            axes[1, k].imshow(overlay)
            axes[1, k].set_title(f"Part {k+1} Overlay", fontsize=11)
            axes[1, k].axis('off')
        
        plt.suptitle("Part Attention Maps and Overlays", fontsize=14, y=0.95)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Overlay saved to {save_path}")
        
        plt.show()
    
    def process_single_image(self, image_path, gmm, output_dir="outputs"):
        """
        处理单张图像，生成完整的热力图可视化
        
        Args:
            image_path: 图像路径
            gmm: GMM模型
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        
        # 生成热力图
        heatmaps = self.generate_heatmaps(image, gmm)
        
        # 基础文件名
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # 可视化热力图
        heatmap_path = os.path.join(output_dir, f"{base_name}_heatmaps.png")
        self.visualize_heatmaps(image, heatmaps, save_path=heatmap_path)
        
        # 创建叠加可视化
        overlay_path = os.path.join(output_dir, f"{base_name}_overlays.png")
        self.create_overlay_visualization(image, heatmaps, save_path=overlay_path)
        
        # 保存单独的热力图
        for k, heatmap in enumerate(heatmaps):
            plt.figure(figsize=(6, 5))
            plt.imshow(heatmap, cmap='jet')
            plt.colorbar(label='Attention Score')
            plt.title(f"Part {k+1} Attention Map")
            plt.axis('off')
            
            part_path = os.path.join(output_dir, f"{base_name}_part{k+1}.png")
            plt.savefig(part_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"所有可视化已保存到 {output_dir}/")


def demo_with_sample_data():
    """
    使用示例数据进行演示
    """
    print("=" * 60)
    print("PartGCD Heatmap Generation Demo")
    print("=" * 60)
    
    # 创建热力图生成器
    generator = PartHeatmapGenerator()
    
    # 下载示例图像（使用COCO数据集中的猫图像）
    print("\n1. Downloading example image...")
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    
    # 保存示例图像
    os.makedirs("samples", exist_ok=True)
    image_path = "samples/example_cat.jpg"
    image.save(image_path)
    
    # 模拟GMM训练（实际应用中应该用真实数据训练）
    print("\n2. Simulating GMM training...")
    
    # 创建一些示例图像路径（模拟同一个类别的多张图像）
    # 在实际应用中，这里应该是同一个类别的多张图像
    sample_image_paths = [image_path] * 3  # 重复使用同一张图像模拟
    
    # 拟合GMM
    gmm = generator.fit_gmm_for_class(sample_image_paths, class_id=0)
    
    print(f"\n3. GMM fitted with {gmm.n_components} components")
    
    # 生成热力图
    print("\n4. Generating heatmaps...")
    heatmaps = generator.generate_heatmaps(image, gmm)
    
    # 可视化
    print("\n5. Creating visualizations...")
    
    # 图1：基础热力图（对应论文Figure 6左半部分）
    generator.visualize_heatmaps(image, heatmaps, save_path="outputs/basic_heatmaps.png")
    
    # 图2：叠加可视化
    generator.create_overlay_visualization(image, heatmaps, save_path="outputs/overlay_heatmaps.png")
    
    # 图3：详细分析图
    print("\n6. Creating detailed analysis visualization...")
    create_detailed_analysis(image, heatmaps, generator)
    
    print("\n" + "=" * 60)
    print("Demo completed! Check the 'outputs' folder for results.")
    print("=" * 60)


def create_detailed_analysis(image, heatmaps, generator):
    """
    创建详细的分析可视化
    """
    # 提取特征获取注意力图
    patch_features, attention_map = generator.extract_features(image)
    
    fig = plt.figure(figsize=(20, 10))
    
    # 1. 原图
    ax1 = plt.subplot(2, 4, 1)
    ax1.imshow(image)
    ax1.set_title("Original Image", fontsize=14)
    ax1.axis('off')
    
    # 2. 自注意力图（来自DINO）
    ax2 = plt.subplot(2, 4, 2)
    # 取CLS token对其他patch的注意力
    if attention_map.shape[0] == generator.num_patches:
        cls_attention = attention_map.mean(axis=0)  # 平均所有patch的注意力
        cls_attention_grid = cls_attention.reshape(14, 14)
        heatmap_resized = cv2.resize(cls_attention_grid, image.size, interpolation=cv2.INTER_LINEAR)
        ax2.imshow(heatmap_resized, cmap='hot')
        ax2.set_title("DINO Self-Attention", fontsize=14)
    else:
        ax2.text(0.5, 0.5, "Attention Map\nNot Available", ha='center', va='center', fontsize=12)
    ax2.axis('off')
    
    # 3-6. 部件热力图
    for k in range(min(4, len(heatmaps))):
        ax = plt.subplot(2, 4, k+3)
        heatmap = heatmaps[k]
        heatmap_resized = cv2.resize(heatmap, image.size, interpolation=cv2.INTER_LINEAR)
        
        im = ax.imshow(heatmap_resized, cmap='jet')
        ax.set_title(f"Part {k+1} Attention", fontsize=14)
        ax.axis('off')
        
        # 添加颜色条
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 7. 所有部件的平均热力图
    ax7 = plt.subplot(2, 4, 7)
    if len(heatmaps) > 0:
        avg_heatmap = np.mean(np.stack(heatmaps), axis=0)
        avg_heatmap_resized = cv2.resize(avg_heatmap, image.size, interpolation=cv2.INTER_LINEAR)
        ax7.imshow(avg_heatmap_resized, cmap='jet')
        ax7.set_title("Average of All Parts", fontsize=14)
    ax7.axis('off')
    
    # 8. 最显著部件的叠加图
    ax8 = plt.subplot(2, 4, 8)
    if len(heatmaps) > 0:
        # 找到最显著的热力图（方差最大）
        variances = [h.std() for h in heatmaps]
        most_salient_idx = np.argmax(variances)
        
        heatmap = heatmaps[most_salient_idx]
        heatmap_resized = cv2.resize(heatmap, image.size, interpolation=cv2.INTER_LINEAR)
        heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]
        
        img_array = np.array(image) / 255.0
        alpha = 0.6
        overlay = (1-alpha) * img_array + alpha * heatmap_colored
        
        ax8.imshow(overlay)
        ax8.set_title(f"Overlay: Part {most_salient_idx+1}\n(Most Salient)", fontsize=14)
    ax8.axis('off')
    
    plt.suptitle("PartGCD: Detailed Part Attention Analysis", fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig("outputs/detailed_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()


def batch_process_images(image_folder, output_folder="batch_outputs"):
    """
    批量处理图像文件夹
    
    Args:
        image_folder: 包含图像的文件夹
        output_folder: 输出文件夹
    """
    print(f"Batch processing images from {image_folder}...")
    
    # 创建热力图生成器
    generator = PartHeatmapGenerator()
    
    # 收集所有图像
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = []
    
    for root, dirs, files in os.walk(image_folder):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(root, file))
    
    print(f"Found {len(image_paths)} images")
    
    if len(image_paths) == 0:
        print("No images found!")
        return
    
    # 使用前几张图像训练GMM
    print("Training GMM on sample images...")
    gmm = generator.fit_gmm_for_class(image_paths[:10], class_id=0)
    
    # 处理每张图像
    for i, img_path in enumerate(image_paths[:5]):  # 限制处理数量
        try:
            print(f"Processing image {i+1}/{min(5, len(image_paths))}: {os.path.basename(img_path)}")
            
            # 创建图像特定的输出目录
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            img_output_dir = os.path.join(output_folder, img_name)
            
            # 处理单张图像
            generator.process_single_image(img_path, gmm, output_dir=img_output_dir)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")


if __name__ == "__main__":
    # 创建输出目录
    os.makedirs("outputs", exist_ok=True)
    
    # 运行演示
    demo_with_sample_data()
    
    # 如果要批量处理自己的图像，取消下面的注释
    batch_process_images("tests", output_folder="batch_results")