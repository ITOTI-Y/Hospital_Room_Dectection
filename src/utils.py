import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from sklearn.neighbors import NearestNeighbors

def visualize_dataset(dataset, num_samples=3, cols=3, figsize=(14, 4)):
    # 获取样本
    samples = [dataset._visualize(i) for i in range(min(num_samples, len(dataset)))]
    
    # 准备图像和标签
    images = [sample[0] for sample in samples]
    labels = [sample[1] for sample in samples]

    # 如果图像是 PyTorch 张量，转换为 numpy 数组
    if isinstance(images[0], torch.Tensor):
        images = [img.permute(1, 2, 0).numpy() for img in images]
    
    # 如果标签是 PyTorch 张量，转换为 numpy 数组
    if isinstance(labels[0], torch.Tensor):
        labels = [label.numpy() for label in labels]

    # 创建子图
    fig, axs = plt.subplots(2, cols, figsize=figsize)
    
    for i, (image, label) in enumerate(zip(images, labels)):
        # 显示图像
        axs[0, i].imshow(image)
        axs[0, i].axis('off')
        axs[0, i].set_title(f'Image {i+1}')
        
        # 显示标签（分割掩码）
        axs[1, i].imshow(label, cmap='nipy_spectral')
        axs[1, i].axis('off')
        axs[1, i].set_title(f'Label {i+1}')
    
    plt.tight_layout()
    plt.show()

def hex_to_rgb(hex_color):
    # Remove the '#' if it's present
    hex_color = hex_color.lstrip('#')
    # Convert hex to RGB
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def process_image_vectorized(image: np.ndarray, color_map: dict):
    # 预处理
    # preprocessed = image
    # preprocessed = preprocess_label_image(image)
    preprocessed = postprocess_image(image)
    # preprocessed = preprocess_image(image)
    
    # 将颜色映射转换为 NumPy 数组
    colors = np.array([list(color) for color in color_map.keys()])
    
    # 创建 NearestNeighbors 对象
    nn = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(colors)
    
    # 重塑图像以便进行向量化操作
    original_shape = preprocessed.shape
    flattened_image = preprocessed.reshape(-1, 3)
    
    # 使用 NearestNeighbors 找到最近的颜色索引
    distances, indices = nn.kneighbors(flattened_image)
    
    # 重塑结果以匹配原始图像形状
    output = indices.reshape(original_shape[:2])

    # 单通道转换为三通道
    # output = np.dstack((output,) * 3)

    # 后处理
    output = postprocess_image(output)
    
    return output[:,:,np.newaxis]

def preprocess_image(image):
    # 转换为Lab颜色空间以更好地处理颜色差异
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
    
    # 应用边缘保持滤波
    smoothed = cv2.edgePreservingFilter(lab, flags=1, sigma_s=60, sigma_r=0.4)
    
    # 增强对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l, a, b = cv2.split(smoothed)
    l = clahe.apply(l)
    enhanced = cv2.merge((l,a,b))
    
    # 转回RGB
    return cv2.cvtColor(enhanced, cv2.COLOR_Lab2RGB)

def postprocess_image(image:np.ndarray):
    image = np.asarray(image, dtype=np.uint8)
    # 应用形态学操作来清理结果
    kernel = np.ones((5,5), np.uint8)
    cleaned = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    return cleaned