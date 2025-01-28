import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from scipy.spatial import KDTree
from src.config import COLOR_MAP, Network_Config
from .node import Node
from .preprocess import preprocess_image, DebugImage, morphology_operation

CONFIG = Network_Config()


class Network:
    def __init__(self, image_path: Image.Image):
        """
        创建节点图，赋值颜色

        Args:
            image (Image.Image): 输入的图像
        """
        self.image = preprocess_image(Image.open(image_path))
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]
        self.color_map = COLOR_MAP
        self.types_map = {v:k for k,v in COLOR_MAP.items()}
        self.graph = nx.Graph()
        self._create_connection_nodes()
        self._create_pedestrian_nodes()
        pass


    def _create_connection_nodes(self):
        """
        创建连接节点
        """
        img_cv = np.array(self.image)
        for conn_type in CONFIG.ROOM_TYPES:
            try:
                conn_type_color = self.types_map[conn_type]
            except KeyError:
                print(f"Color for {conn_type} not found in COLOR_MAP")
                continue
        
            # 创建掩码
            mask = cv2.inRange(img_cv, np.array(conn_type_color), np.array(conn_type_color))
            DebugImage(mask, suffix=conn_type)

            # 形态学操作
            mask = morphology_operation(mask)
            DebugImage(mask, suffix=conn_type)

            # 查找连通组件
            retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
            skewness = np.max(stats[:, 2:4], axis=1) / np.min(stats[:, 2:4], axis=1)
            stats = np.concatenate([stats, skewness.reshape(-1, 1)], axis=1)
            
            stats = stats[stats[:, cv2.CC_STAT_AREA] > CONFIG.AREA_THRESHOLD]
            stats = stats[stats[:, 5] < CONFIG.SKEWNESS]
            if len(stats) <= 1:
                continue
            for i in range(1, retval): # 从1开始，0是背景
                centroid = centroids[i]
                position = (int(centroid[0]), int(centroid[1]))
                node = Node(conn_type, position)
                self.graph.add_node(node)

    def _create_pedestrian_nodes(self):
        """
        创建步行区域节点
        """
        img_cv = np.array(self.image)
        for pedestrian_type in CONFIG.PEDESTRIAN_TYPES:
            pedestrian_type_color = self.types_map[pedestrian_type]
            mask = cv2.inRange(img_cv, np.array(pedestrian_type_color), np.array(pedestrian_type_color))
            mask = morphology_operation(mask)
            retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

            for i in range(1, retval):
                area = stats[i, cv2.CC_STAT_AREA]
                if area < CONFIG.AREA_THRESHOLD:
                    continue

                x, y, w, h = stats[i, :4]

                # 生成网格点坐标
                gx = np.arange(x, x+w, CONFIG.GRID_SIZE)
                gy = np.arange(y, y+h, CONFIG.GRID_SIZE)
                grid_x, grid_y = np.meshgrid(gx, gy)
                grid_x = grid_x.flatten()
                grid_y = grid_y.flatten()

                # 筛选位于步行区域内的网格点
                valid_indices = mask[grid_y, grid_x] != 0
                valid_x = grid_x[valid_indices]
                valid_y = grid_y[valid_indices]

                # 创建节点并添加到图中
                for x, y in zip(valid_x, valid_y):
                    node = Node(pedestrian_type, (int(x), int(y)))
                    self.graph.add_node(node)


    def plot(self, save: bool = False):
        """
        可视化节点图
        """
        # 将图片尺寸从像素转换为英寸（考虑 DPI）
        dpi = 100
        width_inches = self.width / dpi
        height_inches = self.height / dpi
        
        # 创建指定尺寸的图形
        fig = plt.figure(figsize=(width_inches, height_inches), dpi=dpi)
        # 移除图形周围的空白边距
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        
        # 显示图像和节点
        ax.imshow(self.image)
        pos = {node:node.pos for node in self.graph.nodes}
        labels = {node:node.type for node in self.graph.nodes}
        nx.draw(self.graph, pos, labels=labels, with_labels=True, 
                node_size=50, node_color='red', font_size=8, 
                font_weight='bold', ax=ax)
        
        if save:
            plt.savefig('./debug/network.png', 
                        dpi=dpi, 
                        bbox_inches='tight',
                        pad_inches=0)
            plt.close()
        else:
            plt.show()
