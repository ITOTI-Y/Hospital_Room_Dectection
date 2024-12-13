import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from scipy.spatial import KDTree
from src.config import COLOR_MAP, Network_Config
from .node import Node

CONFIG = Network_Config()


class Network:
    def __init__(self, image: Image.Image):
        """
        创建节点图，赋值颜色

        Args:
            image (Image.Image): 输入的图像
        """
        self.image = image
        self.width = image.width
        self.height = image.height
        self.color_map = COLOR_MAP
        self.types_map = {v:k for k,v in COLOR_MAP.items()}
        self.graph = nx.Graph()
        self._porcess_image_colors()
        self._create_connection_nodes()
        self._create_pedestrian_nodes()

    def _porcess_image_colors(self):
        """
        将每个像素的颜色更改为COLOR_MAP中与之最接近的颜色
        """
        img_array = np.array(self.image)
        pixels = img_array.reshape(-1, 3)
        kdtree = KDTree(list(self.color_map.keys()))
        _, closest_indices = kdtree.query(pixels)
        color_values = np.array(list(self.color_map.keys()))
        new_pixels = color_values[closest_indices]
        new_img_array = new_pixels.reshape(img_array.shape).astype(np.uint8)
        self.image = Image.fromarray(new_img_array)

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

            # 查找连通组件
            retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

            for i in range(1, retval): # 从1开始，0是背景
                area = stats[i, cv2.CC_STAT_AREA]
                if area < CONFIG.AREA_THRESHOLD:
                    continue
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


    def plot(self):
        """
        可视化节点图
        """
        plt.imshow(self.image)
        pos = {node:node.pos for node in self.graph.nodes}
        labels = {node:node.type for node in self.graph.nodes}
        nx.draw(self.graph, pos, labels=labels, with_labels=True, node_size=50, node_color='red', font_size=8, font_weight='bold')
        plt.show()
