import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from src.config import COLOR_MAP

class NodeGraph:
    def __init__(self, image: Image.Image):
        """
        创建节点图，赋值颜色

        Args:
            image (Image.Image): 输入的图像
        """
        self.image = image
        self.width = image.width
        self.height = image.height
        self.node_width = 100
        self.node_height = int((100 * self.height / self.width))
        self.aspect_ratio = self.width / self.node_width
        self.graph = nx.grid_2d_graph(self.node_width, self.node_height)
        self.pos = self._scale_position()
        self._sample_colors_from_image()

    def _scale_position(self):
        """
        根据 aspect_ratio 缩放节点位置

        Returns:
            dict: 缩放后的节点位置
        """
        pos = {node: (node[0], node[1]) for node in self.graph.nodes()}
        scaled_pos = {node: (x * self.aspect_ratio, y * self.aspect_ratio) for node, (x, y) in pos.items()}
        return scaled_pos
    
    def _sample_colors_from_image(self):
        """
        从图像中采样颜色
        """
        image_array = np.array(self.image)

        for node in self.graph.nodes():
            x, y = int(node[0] * self.aspect_ratio), int(node[1] * self.aspect_ratio)
            color = image_array[y, x]
            self.graph.nodes[node]['types'] = self._find_closest_color(color)

    def _find_closest_color(self, color: np.ndarray):
        """
        返回颜色对应的类别

        Args:
            color (np.ndarray): 颜色
        """
        color_array = np.array(list(COLOR_MAP.keys()))
        distances = np.linalg.norm(color_array - color, axis=1)
        closest_index = np.argmin(distances)
        result = tuple(color_array[closest_index])

        return result
    
    def draw_graph(self):
        """
        可视化节点图
        """
        plt.imshow(self.image)
        nx.draw(self.graph, self.pos, with_labels=False, node_size=10, node_color='red', edge_color='black')
        plt.show()