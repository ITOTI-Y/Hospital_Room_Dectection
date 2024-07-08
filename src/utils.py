import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from sklearn.neighbors import NearestNeighbors

def visualize_dataset(dataset, num_samples=3, cols=3, figsize=(14, 4)):
    # 获取样本
    samples = [dataset[i] for i in range(min(num_samples, len(dataset)))]
    
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

def find_nearest_color(pixel, colors, n_neighbors=1):
    distances, indices = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean').fit(colors).kneighbors([pixel])
    return colors[indices[0][0]]

def process_image(image:np.array, color_map:dict):
    colors = np.array([list(color) for color in color_map.keys()])
    output = np.zeros(image.shape[:2], dtype=object)
    plt.imshow(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel = image[i, j]
            output[i, j] = find_nearest_color(pixel, colors)
    pass