import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
import os
import plotly.graph_objects as go
from pathlib import Path
from PIL import Image
from scipy.spatial import KDTree, distance
from src.config import COLOR_MAP, Network_Config
from .node import Node
from .preprocess import preprocess_image, DebugImage, morphology_operation

CONFIG = Network_Config()


class Network:
    def __init__(self):
        """
        创建节点图，赋值颜色

        Args:
            image (Image.Image): 输入的图像
        """
        self.color_map = COLOR_MAP
        self.types_map = {v['name']: k for k, v in COLOR_MAP.items()}

    def add_node(self, node: Node):
        """
        添加节点到图中

        Args:
            node (Node): 待添加的节点
        """
        self.graph.add_node(node)
        self.id_to_node[node.id] = node

    def get_node_by_id(self, node_id: int) -> Node:
        """
        通过节点 ID 获取节点

        Args:
            node_id (int): 节点 ID

        Returns:
            Node: 获取到的节点
        """
        return self.id_to_node[node_id]

    def connect_nodes_by_ids(self, node_id1: int, node_id2: int):
        """
        通过节点 ID 连接两个节点

        Args:
            node_id1 (int): 节点1 ID
            node_id2 (int): 节点2 ID
        """
        node1 = self.get_node_by_id(node_id1)
        node2 = self.get_node_by_id(node_id2)
        if node1 and node2:
            self.graph.add_edge(node1, node2)

    def _create_outside_mask(self):
        """
        创建外部节点
        """
        img = self.image.copy()
        for conn_type in CONFIG.OUTSIDE_TYPES:
            try:
                conn_type_color = self.types_map[conn_type]
            except KeyError:
                print(f"Color for {conn_type} not found in COLOR_MAP")
                continue

            # 创建掩码,提取对应区域
            mask = cv2.inRange(img, np.array(conn_type_color),
                               np.array(conn_type_color))

            # 形态学操作
            mask = morphology_operation(mask)

            # 查找连通组件
            self.id_map[(mask != 0)] = CONFIG.OUTSIDE_ID

    def _create_room_nodes(self, zlevel: int):
        """
        Create Room Nodes
        """
        img = self.image.copy()
        for conn_type in CONFIG.ROOM_TYPES:
            try:
                conn_type_color = self.types_map[conn_type]
            except KeyError:
                print(f"Color for {conn_type} not found in COLOR_MAP")
                continue

            # 创建掩码,提取对应区域
            mask = cv2.inRange(img, np.array(conn_type_color),
                               np.array(conn_type_color))

            # 形态学操作
            mask = morphology_operation(mask)

            # 查找连通组件
            retval, labels, stats, centroids = cv2.connectedComponentsWithStats(
                mask, connectivity=4)
            if retval <= 1:
                continue
            for i in range(1, retval):
                centroid = centroids[i]
                position = (int(centroid[0]), int(centroid[1]), zlevel)
                node = Node(conn_type, position)
                node.time = COLOR_MAP[conn_type_color]['time']
                self.add_node(node)
                self.id_map[(labels == i)] = node.id
    
    def _create_vertical_nodes(self, zlevel: int):
        """
        Create Room Nodes
        """
        img = self.image.copy()
        for conn_type in CONFIG.VERTICAL_TYPES:
            try:
                conn_type_color = self.types_map[conn_type]
            except KeyError:
                print(f"Color for {conn_type} not found in COLOR_MAP")
                continue

            # 创建掩码,提取对应区域
            mask = cv2.inRange(img, np.array(conn_type_color),
                               np.array(conn_type_color))

            # 形态学操作
            mask = morphology_operation(mask)

            # 查找连通组件
            retval, labels, stats, centroids = cv2.connectedComponentsWithStats(
                mask, connectivity=4)
            if retval <= 1:
                continue
            for i in range(1, retval):
                centroid = centroids[i]
                position = (int(centroid[0]), int(centroid[1]), zlevel)
                node = Node(conn_type, position)
                node.time = COLOR_MAP[conn_type_color]['time']
                self.add_node(node)
                self.id_map[(labels == i)] = node.id

    def _create_connection_nodes(self, zlevel: int):
        """
        创建连接节点
        """
        img = self.image.copy()
        pass_id = [CONFIG.OUTSIDE_ID,
                   CONFIG.PEDESTRIAN_ID, CONFIG.BACKGROUND_ID]
        dilated_masks = []
        for conn_type in CONFIG.CONNECTION_TYPES:
            try:
                conn_type_color = self.types_map[conn_type]
            except KeyError:
                print(f"Color for {conn_type} not found in COLOR_MAP")
                continue

            # 创建掩码,提取对应区域
            mask = cv2.inRange(img, np.array(conn_type_color),
                               np.array(conn_type_color))

            # 形态学操作
            mask = morphology_operation(mask)

            # 膨胀核
            kernel = np.ones((3, 3), np.uint8)

            # 查找连通组件
            retval, labels, stats, centroids = cv2.connectedComponentsWithStats(
                mask, connectivity=4)
            if retval <= 1:
                continue
            for i in range(1, retval):
                centroid = centroids[i]
                position = (int(centroid[0]), int(centroid[1]), zlevel)
                node = Node(conn_type, position)
                node.time = CONFIG.CONNECTION_TIME
                self.add_node(node)
                component_mask = (labels == i).astype(np.uint8) * 255
                self.id_map[(component_mask != 0)] = node.id
                dilated_component_mask = cv2.dilate(
                    component_mask, kernel, iterations=1)
                connection_ids = np.unique(
                    self.id_map[(dilated_component_mask != 0)])
                if CONFIG.OUTSIDE_ID in connection_ids:
                    node.door = 'out'
                elif CONFIG.PEDESTRIAN_ID in connection_ids:
                    node.door = 'in'
                else:
                    node.door = 'room'
                for connection_id in connection_ids:
                    if connection_id != node.id and connection_id not in pass_id:  # -1为外部id
                        self.connect_nodes_by_ids(node.id, connection_id)

    def _create_pedestrian_nodes(self, zlevel: int):
        """
        创建步行区域节点
        """
        img_cv = np.array(self.image)
        for region_type in CONFIG.PEDESTRIAN_TYPES:
            pedestrian_type_color = np.array(self.types_map[region_type])

            # 创建掩码,提取对应区域
            mask = cv2.inRange(img_cv, pedestrian_type_color,
                               pedestrian_type_color)
            mask = morphology_operation(mask)
            self.id_map[(mask != 0)] = CONFIG.PEDESTRIAN_ID
            self._create_mesh_nodes(mask, region_type, zlevel=zlevel)

    def _create_outside_nodes(self, zlevel: int):
        """
        创建外部区域节点
        """
        img_cv = np.array(self.image)
        for region_type in CONFIG.OUTSIDE_TYPES:
            outside_type_color = np.array(self.types_map[region_type])

            # 创建掩码,提取对应区域
            mask = cv2.inRange(img_cv, outside_type_color, outside_type_color)
            mask = morphology_operation(mask)
            self.id_map[(mask != 0)] = CONFIG.OUTSIDE_ID
            self._create_mesh_nodes(
                mask, region_type, times=CONFIG.OUTSIDE_TIMES, zlevel=zlevel)

    def _create_mesh_nodes(self, mask: np.array, region_type: str, times: int = 1, zlevel: int = 0):
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8)
        grid_size = CONFIG.GRID_SIZE * times

        for i in range(1, retval):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < CONFIG.AREA_THRESHOLD:  # 过滤掉面积小于阈值的区域
                continue

            x, y, w, h, _ = stats[i]

            # 创建稠密网格
            gx = np.arange(x, x+w, grid_size)
            gy = np.arange(y, y+h, grid_size)
            grid_x, grid_y = np.meshgrid(gx, gy)

            # 将稠密网格转换为稀疏网格
            valid_mask = labels[grid_y, grid_x] == i
            valid_x = grid_x[valid_mask]
            valid_y = grid_y[valid_mask]

            # 创建节点
            positions = np.stack(
                [valid_x, valid_y, np.full(valid_x.shape, zlevel)], axis=1)
            nodes = []
            for pos in positions:
                node = Node(region_type, tuple(pos))
                node.time = CONFIG.PEDESTRAIN_TIME
                self.add_node(node)
                self.id_to_node[node.id] = node
                nodes.append(node)

            if not nodes:
                continue

            # 创建 KDTree
            node_positions = np.array([node.pos for node in nodes])
            tree = KDTree(node_positions)

            # 最大连接距离
            max_distance = 1.1 * grid_size

            # 连接节点
            for i, node in enumerate(nodes):
                distances, indices = tree.query(node.pos, k=9)  # 除自身外的8个最近邻
                for dis, j in zip(distances, indices):
                    if i == j or dis > max_distance:
                        continue
                    neighbor = nodes[j]

                    # 计算节点之间的距离
                    # dis = distance.euclidean(node.pos, neighbor.pos)
                    tolerance = grid_size / 4
                    if (
                        (abs(node.pos[0] - neighbor.pos[0]) < tolerance and abs(node.pos[1] - neighbor.pos[1]) > tolerance) or
                        (abs(node.pos[1] - neighbor.pos[1]) <
                         tolerance and abs(node.pos[0] - neighbor.pos[0]) > tolerance)
                    ):
                        self.connect_nodes_by_ids(node.id, neighbor.id)

    def _connect_pedestrian_connection(self, outside: bool = False):
        """
        连接房间和门节点
        """
        connection_nodes = [
            node for node in self.graph.nodes if node.type in CONFIG.CONNECTION_TYPES]
        pedestrian_nodes = [
            node for node in self.graph.nodes if node.type in CONFIG.PEDESTRIAN_TYPES]
        outside_nodes = [
            node for node in self.graph.nodes if node.type in CONFIG.OUTSIDE_TYPES]
        if pedestrian_nodes:
            pe_tree = KDTree([node.pos for node in pedestrian_nodes])
        else:
            return
        if outside_nodes:
            out_tree = KDTree([node.pos for node in outside_nodes])
        max_distance = 1.2 * CONFIG.GRID_SIZE
        for conn_node in connection_nodes:
            if conn_node.door == 'room':
                continue
            else:
                pe_distances, pe_indices = pe_tree.query(conn_node.pos, k=1)
                if outside:
                    out_distances, out_indices = out_tree.query(
                        conn_node.pos, k=1)
                    if conn_node.door == 'out':
                        nearest_outside = outside_nodes[out_indices]
                        self.connect_nodes_by_ids(
                            conn_node.id, nearest_outside.id)
                        if pe_distances < max_distance:
                            nearest_pedestrian = pedestrian_nodes[pe_indices]
                            self.connect_nodes_by_ids(
                                conn_node.id, nearest_pedestrian.id)
                else:
                    nearest_pedestrian = pedestrian_nodes[pe_indices]
                    self.connect_nodes_by_ids(
                        conn_node.id, nearest_pedestrian.id)

    def run(self, image_path: str, zlevel: int = 0, outside: bool = False):
        """
        运行节点图
        """
        self.graph = nx.Graph()
        self.id_to_node = {}
        self.image = np.asarray(Image.open(image_path).rotate(CONFIG.IMAGE_ROTATE))
        self.id_map = np.zeros_like(self.image, dtype=int)
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]
        self._create_outside_mask()
        self._create_room_nodes(zlevel=zlevel)
        self._create_vertical_nodes(zlevel=zlevel)
        self._create_pedestrian_nodes(zlevel=zlevel)
        if outside:
            self._create_outside_nodes(zlevel=zlevel)
        self._create_connection_nodes(zlevel=zlevel)
        self._connect_pedestrian_connection(outside=outside)
        return self.graph

    def plot(self, save: bool = False):
        """
        可视化节点图 (3D)
        """
        dpi = 100
        width_inches = self.width / dpi
        height_inches = self.height / dpi
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

        fig = plt.figure(figsize=(width_inches, height_inches), dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')  # 创建 3D 子图
        # ax.set_axis_off() # 3D绘图时，通常不隐藏坐标轴

        # 注意：在 3D 绘图中，通常不直接显示背景图像。
        #       你可以在这里添加一些代码来表示楼层的平面。

        # 获取节点坐标和标签
        pos = {node: node.pos for node in self.graph.nodes()}
        labels = {node: f"{node.type}" for node in self.graph.nodes()
                  }  # 3D显示时,标签通常简单些

        # 提取 x, y, z 坐标
        x = [pos[node][0] for node in self.graph.nodes()]
        y = [pos[node][1] for node in self.graph.nodes()]
        z = [pos[node][2] for node in self.graph.nodes()]
        
        # 根据节点类型获取节点大小
        node_sizes = []
        for node in self.graph.nodes():
            if node.type in CONFIG.PEDESTRIAN_TYPES:
                node_sizes.append(CONFIG.NODE_SIZE_PEDESTRIAN * 10)  # 放大倍数，使其在matplotlib中可见
            elif node.type in CONFIG.CONNECTION_TYPES:
                node_sizes.append(CONFIG.NODE_SIZE_CONNECTION * 10)
            elif node.type in CONFIG.VERTICAL_TYPES:
                node_sizes.append(CONFIG.NODE_SIZE_VERTICAL * 10)
            elif node.type in CONFIG.ROOM_TYPES:
                node_sizes.append(CONFIG.NODE_SIZE_ROOM * 10)
            elif node.type in CONFIG.OUTSIDE_TYPES:
                node_sizes.append(CONFIG.NODE_SIZE_OUTSIDE * 10)
            else:
                node_sizes.append(CONFIG.NODE_SIZE_DEFAULT * 10)

        # 绘制节点
        ax.scatter(x, y, z, c='red', s=node_sizes, alpha=CONFIG.NODE_OPACITY)  # 使用可变大小和统一透明度的节点

        # 绘制边 (需要遍历边列表)
        for edge in self.graph.edges():
            x_coords = [pos[edge[0]][0], pos[edge[1]][0]]
            y_coords = [pos[edge[0]][1], pos[edge[1]][1]]
            z_coords = [pos[edge[0]][2], pos[edge[1]][2]]
            ax.plot(x_coords, y_coords, z_coords, c='blue',
                    linewidth=0.5)  # 使用 plot 绘制线段

        # 给节点添加文本
        for node in self.graph.nodes:
            ax.text(pos[node][0], pos[node][1], pos[node]
                    [2], labels[node], fontsize=8)

        # 设置坐标轴标签
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        if save:
            plt.savefig('./debug/network_3d.png', dpi=dpi,
                        bbox_inches='tight', pad_inches=0)
            plt.close()
        else:
            plt.show()

    def plot_plotly(self, save: bool = False):
        pos = {node: node.pos for node in self.graph.nodes()}
        
        # 按类型分组节点
        node_groups = {}
        for node_type in set([node.type for node in self.graph.nodes()]):
            node_groups[node_type] = {
                'x': [], 'y': [], 'z': [], 'text': [], 'sizes': []
            }
        
        # 填充分组数据
        for node in self.graph.nodes():
            x, y, z = pos[node]
            node_groups[node.type]['x'].append(self.width - x if CONFIG.IMAGE_MIRROR else x)
            node_groups[node.type]['y'].append(y)
            node_groups[node.type]['z'].append(z)
            
            # 根据配置决定是否显示人行区域节点的标签
            if node.type in CONFIG.PEDESTRIAN_TYPES and not CONFIG.SHOW_PEDESTRIAN_LABELS:
                node_groups[node.type]['text'].append("")
            else:
                node_groups[node.type]['text'].append(f"{node.type}")
            
            # 根据节点类型设置大小
            if node.type in CONFIG.PEDESTRIAN_TYPES:
                node_groups[node.type]['sizes'].append(CONFIG.NODE_SIZE_PEDESTRIAN)
            elif node.type in CONFIG.CONNECTION_TYPES:
                node_groups[node.type]['sizes'].append(CONFIG.NODE_SIZE_CONNECTION)
            elif node.type in CONFIG.VERTICAL_TYPES:
                node_groups[node.type]['sizes'].append(CONFIG.NODE_SIZE_VERTICAL)
            elif node.type in CONFIG.ROOM_TYPES:
                node_groups[node.type]['sizes'].append(CONFIG.NODE_SIZE_ROOM)
            elif node.type in CONFIG.OUTSIDE_TYPES:
                node_groups[node.type]['sizes'].append(CONFIG.NODE_SIZE_OUTSIDE)
            else:
                node_groups[node.type]['sizes'].append(CONFIG.NODE_SIZE_DEFAULT)
        
        # 分类连接线（水平和垂直）
        horizontal_edges = {'x': [], 'y': [], 'z': []}
        vertical_edges = {'x': [], 'y': [], 'z': []}
        
        for edge in self.graph.edges():
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            
            # 判断是否为垂直连接线（通过检查z坐标是否相同）
            if abs(z0 - z1) > 0.01:  # 允许一点误差
                if CONFIG.IMAGE_MIRROR:
                    vertical_edges['x'].extend([self.width - x0, self.width - x1, None])
                else:
                    vertical_edges['x'].extend([x0, x1, None])
                vertical_edges['y'].extend([y0, y1, None])
                vertical_edges['z'].extend([z0, z1, None])
            else:
                if CONFIG.IMAGE_MIRROR:
                    horizontal_edges['x'].extend([self.width - x0, self.width - x1, None])
                else:
                    horizontal_edges['x'].extend([x0, x1, None])
                horizontal_edges['y'].extend([y0, y1, None])
                horizontal_edges['z'].extend([z0, z1, None])
        
        # 创建水平连接线轨迹
        horizontal_edge_trace = go.Scatter3d(
            x=horizontal_edges['x'],
            y=horizontal_edges['y'],
            z=horizontal_edges['z'],
            line=dict(width=CONFIG.EDGE_WIDTH, color=CONFIG.HORIZONTAL_EDGE_COLOR),
            hoverinfo='none',
            mode='lines',
            name='水平连接'  # 添加图例名称
        )
        
        # 创建垂直连接线轨迹
        vertical_edge_trace = go.Scatter3d(
            x=vertical_edges['x'],
            y=vertical_edges['y'],
            z=vertical_edges['z'],
            line=dict(width=CONFIG.EDGE_WIDTH, color=CONFIG.VERTICAL_EDGE_COLOR),
            hoverinfo='none',
            mode='lines',
            name='垂直连接'  # 添加图例名称
        )
        
        # 为每种节点类型创建轨迹
        node_traces = []
        for node_type, data in node_groups.items():
            if not data['x']:  # 跳过空分组
                continue
                
            node_trace = go.Scatter3d(
                x=data['x'],
                y=data['y'],
                z=data['z'],
                mode='markers+text',
                hoverinfo='text',
                name=node_type,  # 添加图例名称
                marker=dict(
                    size=data['sizes'],
                    sizemode='diameter',  # 确保缩放时节点大小保持不变
                    opacity=CONFIG.NODE_OPACITY,
                    color=self.get_color_for_type(node_type),  # 使用类型对应的颜色
                    line_width=2
                ),
                text=data['text'],
                textposition="top center"
            )
            node_traces.append(node_trace)
        
        # 创建图形
        fig = go.Figure(
            data=[horizontal_edge_trace, vertical_edge_trace] + node_traces,
            layout=go.Layout(
                title='2D Network Graph',
                showlegend=True,  # 显示图例
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                legend=dict(
                    itemsizing='constant',  # 图例项大小一致
                    font=dict(size=10),  # 图例文字大小
                ),
                scene=dict(
                    xaxis=dict(title='X'),
                    yaxis=dict(title='Y'),
                    zaxis=dict(title='Z'),
                )
            )
        )

        if save:
            fig.write_html(CONFIG.RESULT_PATH / '2D_network_plotly.html')  # 保存为 HTML
        else:
            fig.show()
            
        return fig
        
    def get_color_for_type(self, node_type):
        """根据节点类型获取对应的颜色"""
        if not CONFIG.NODE_COLOR:
            return '#1f77b4'
        try:
            # 尝试从COLOR_MAP获取颜色
            for color_tuple, info in COLOR_MAP.items():
                if info['name'] == node_type:
                    # 将RGB元组转为十六进制颜色
                    return f'rgb{color_tuple}'
        except:
            pass
        # 默认颜色
        return '#1f77b4'

class SuperNetwork:
    def __init__(self, tolerance:int = None, base_floor:int = 0, floor_height:int = 10):
        """
        初始化SuperNetwork对象
        
        Args:
            tolerance (int, optional): 不同楼层垂直连接的容差值，如果为None则自动计算
            base_floor (int, optional): 起始楼层编号，默认为0
            floor_height (int, optional): 楼层间的默认高度，默认为10
        """
        self.super_graph = nx.Graph()
        self.vertical_connection_types = CONFIG.VERTICAL_TYPES
        self.tolerance = tolerance
        self.networks = []
        self.base_floor = base_floor
        self.floor_height = floor_height
        self.floor_map = {}  # 用于存储楼层和对应的z坐标
        self.width = None
        self.height = None
    
    def add_network(self, network:nx.Graph):
        self.networks.append(network)
        self.super_graph.add_nodes_from(network.nodes)
        self.super_graph.add_edges_from(network.edges)

    def detect_floor_from_filename(self, filename):
        """
        从文件名中检测楼层信息
        
        Args:
            filename (str): 图像文件名
            
        Returns:
            int: 检测到的楼层编号，如果无法检测则返回None
        """
        import re
        # 尝试匹配常见的楼层表示方式，如"1F"、"B1"、"L2"等
        floor_patterns = {
            r'-(\d+)[Ff]': lambda x: -int(x),  # 匹配 -1F, -2f 等(负楼层)
            r'(\d+)[Ff]': lambda x: int(x),  # 匹配 1F, 2f 等
            r'[Bb](\d+)': lambda x: -int(x),  # 匹配 B1, b2 等(负楼层)
            r'[Ll](\d+)': lambda x: int(x),  # 匹配 L1, l2 等
            r'[Ff](\d+)': lambda x: int(x),  # 匹配 F1, f2 等
        }
        
        # 确保正确处理文件路径，无论是正斜杠还是反斜杠
        basename = os.path.basename(filename.replace('\\', '/'))
        for pattern, convert in floor_patterns.items():
            match = re.search(pattern, basename)
            if match:
                return convert(match.group(1))
        return None
    
    def auto_detect_floors(self, image_paths):
        """
        自动检测图像文件对应的楼层
        
        Args:
            image_paths (list): 图像文件路径列表
            
        Returns:
            dict: 文件路径到楼层编号的映射
        """
        floors = {}
        
        # 尝试从文件名检测楼层信息
        for path in image_paths:
            floor = self.detect_floor_from_filename(path)
            if floor is not None:
                floors[path] = floor
        
        # 如果没有从所有文件中检测到楼层信息，则按顺序分配楼层
        if len(floors) < len(image_paths):
            unassigned_paths = [p for p in image_paths if p not in floors]
            # 如果没有检测到任何楼层，从base_floor开始
            start_floor = self.base_floor if not floors else max(floors.values()) + 1
            for i, path in enumerate(unassigned_paths):
                floors[path] = start_floor + i
        
        return floors
    
    def calculate_floor_z_levels(self, floor_to_path):
        """
        计算每个楼层对应的z坐标
        
        Args:
            floor_to_path (dict): 楼层编号到文件路径的映射
            
        Returns:
            dict: 楼层编号到z坐标的映射
        """
        # 按照楼层编号从小到大排序
        sorted_floors = sorted(floor_to_path.keys())
        
        # 计算相邻楼层之间的高度差
        if len(sorted_floors) <= 1:
            # 如果只有一个楼层，使用默认高度
            z_levels = {sorted_floors[0]: sorted_floors[0] * self.floor_height}
        else:
            # 使用相等的高度差
            z_levels = {floor: floor * self.floor_height for floor in sorted_floors}
        
        return z_levels
    
    def auto_calculate_tolerance(self):
        """
        自动计算不同楼层垂直连接的容差值
        
        Returns:
            int: 计算得到的容差值
        """
        # 获取所有垂直连接节点的位置
        vertical_nodes = [node for node in self.super_graph.nodes if node.type in self.vertical_connection_types]
        
        if not vertical_nodes or len(vertical_nodes) < 2:
            # 如果没有足够的垂直连接节点，使用默认值
            return 30
        
        # 计算垂直连接节点的平均间距
        positions = np.array([node.pos[:2] for node in vertical_nodes])  # 只考虑x,y坐标
        
        # 使用KDTree计算每个点到最近点的距离
        tree = KDTree(positions)
        distances, _ = tree.query(positions, k=2)  # k=2表示每个点自身和最近的另一个点
        
        # 取第二列(最近点的距离)，计算均值作为容差基准
        avg_distance = np.mean(distances[:, 1])
        
        # 添加一个安全系数(例如1.2)，确保能够正确连接
        return int(avg_distance * 1.2)
    
    def connect_floors(self):
        """
        连接不同楼层之间的垂直交通节点，确保仅连接相同类型的节点
        """
        all_vertical_nodes = [node for node in self.super_graph.nodes if node.type in self.vertical_connection_types]
        if not all_vertical_nodes:
            return
        
        # 如果tolerance未指定，自动计算
        if self.tolerance is None:
            self.tolerance = self.auto_calculate_tolerance()
            print(f"自动计算的楼层连接容差: {self.tolerance}")
        
        # 按照节点类型分组
        type_to_nodes = {}
        for node in all_vertical_nodes:
            if node.type not in type_to_nodes:
                type_to_nodes[node.type] = []
            type_to_nodes[node.type].append(node)
        
        # 对每种类型的节点建立KDTree
        connected_pairs = set()  # 用于记录已连接的节点对
        
        # 为每种垂直交通节点类型分别构建连接
        for node_type, nodes in type_to_nodes.items():
            if len(nodes) < 2:
                continue
                
            # 构建KDTree，仅使用x和y坐标
            positions_xy = np.array([node.pos[:2] for node in nodes])
            tree = KDTree(positions_xy)
            
            # 为每个节点查找可能的垂直连接
            for i, node in enumerate(nodes):
                # 查询容差范围内的所有点
                indices = tree.query_ball_point(node.pos[:2], r=self.tolerance)
                
                for idx in indices:
                    target_node = nodes[idx]
                    # 避免自连接
                    if target_node == node:
                        continue
                        
                    # 确保不同高度的节点才连接
                    if abs(target_node.pos[2] - node.pos[2]) < 0.01:  # 允许一点误差
                        continue
                        
                    # 创建唯一标识这对节点的键
                    node_pair = tuple(sorted([hash(node), hash(target_node)]))
                    if node_pair not in connected_pairs:
                        self.super_graph.add_edge(node, target_node)
                        connected_pairs.add(node_pair)
        
        print(f"已连接 {len(connected_pairs)} 对垂直交通节点")

    def run(self, image_paths: list, zlevels: list = None):
        """
        运行多楼层网络构建
        
        Args:
            image_paths (list): 图像文件路径列表
            zlevels (list, optional): 每个图像对应的z坐标，如果为None则自动计算
        """
        if zlevels is None:
            # 自动计算楼层和Z坐标
            floor_to_path = self.auto_detect_floors(image_paths)
            path_to_floor = {v: k for k, v in floor_to_path.items()}
            
            # 计算每个楼层的z坐标
            self.floor_map = self.calculate_floor_z_levels(path_to_floor)
            
            # 按照image_paths顺序获取对应的z坐标
            zlevels = [v for k,v in self.floor_map.items()]

            print("自动计算的楼层z坐标:")
            for path, z in zip(image_paths, zlevels):
                print(f"  {os.path.basename(path)}: z = {z}")
        else:
            if len(image_paths) != len(zlevels):
                raise ValueError("image_paths和zlevels的长度必须相同")
        
        # 构建各楼层网络
        for image_path, zlevel in zip(image_paths, zlevels):
            network = Network()
            self.add_network(network.run(image_path, zlevel=zlevel)) # 默认不连接室外节点
            if self.width is None and self.height is None:
                self.width = network.width
                self.height = network.height
            else:
                if self.width != network.width or self.height != network.height:
                    raise ValueError("所有图片的宽度和高必须相同")
        self.connect_floors()
        return self.super_graph

    def _create_floor_selection_controls(self, all_z_levels, min_z, max_z):
        """
        创建楼层选择控件
        
        Args:
            all_z_levels (list): 所有z坐标
            min_z (float): 最小z坐标
            max_z (float): 最大z坐标
            
        Returns:
            dict: 控件配置
        """
        # 创建楼层标签映射
        floor_labels = {}
        
        # 如果存在floor_map，使用它来显示实际楼层号
        if hasattr(self, 'floor_map') and self.floor_map:
            for z in all_z_levels:
                # 找到最接近z值的楼层编号
                closest_floor = min(self.floor_map.items(), key=lambda x: abs(x[1] - z))[0]
                floor_prefix = "B" if closest_floor < 0 else ""
                floor_num = abs(closest_floor)
                floor_labels[z] = f"{floor_prefix}{floor_num}F"
        else:
            # 否则使用z坐标作为标签
            for z in all_z_levels:
                floor_labels[z] = f"Z={z}"
        
        sliders = [
            dict(
                active=0,
                currentvalue={"prefix": "当前楼层: "},
                pad={"t": 50},
                steps=[
                    dict(
                        label=floor_labels[z_level],
                        method="relayout",
                        args=[
                            {"scene.zaxis.range": [z_level - 0.5, z_level + 0.5]}
                        ]
                    ) for z_level in all_z_levels
                ],
                name="Z轴楼层选择"
            ),
            dict(
                active=len(all_z_levels),
                currentvalue={"prefix": "显示全部楼层: "},
                pad={"t": 100},
                steps=[
                    dict(
                        label="全部",
                        method="relayout",
                        args=[
                            {"scene.zaxis.range": [min_z - 5, max_z + 5]}
                        ]
                    )
                ],
                name="显示全部楼层"
            )
        ]
        
        return {"sliders": sliders}
        
    def plot_plotly(self, save: bool = False):
        """
        使用Plotly绘制3D网络图
        
        Args:
            save (bool, optional): 是否保存图像文件. Defaults to False.
            
        Returns:
            plotly.graph_objects.Figure: Plotly图形对象
        """
        pos = {node: node.pos for node in self.super_graph.nodes()}
        
        # 获取所有独特的Z坐标值（楼层）
        all_z_levels = sorted(list(set([node.pos[2] for node in self.super_graph.nodes()])))
        min_z, max_z = min(all_z_levels), max(all_z_levels)
        
        # 按类型分组节点
        node_groups = {}
        for node_type in set([node.type for node in self.super_graph.nodes()]):
            node_groups[node_type] = {
                'x': [], 'y': [], 'z': [], 'text': [], 'sizes': []
            }
        
        # 填充分组数据
        for node in self.super_graph.nodes():
            x, y, z = pos[node]
            if CONFIG.IMAGE_MIRROR:
                x = self.width - x
            node_groups[node.type]['x'].append(x)
            node_groups[node.type]['y'].append(y)
            node_groups[node.type]['z'].append(z)
            
            # 根据配置决定是否显示人行区域节点的标签
            if node.type in CONFIG.PEDESTRIAN_TYPES and not CONFIG.SHOW_PEDESTRIAN_LABELS:
                node_groups[node.type]['text'].append("")
            else:
                node_groups[node.type]['text'].append(f"{node.type}")
            
            # 根据节点类型设置大小
            if node.type in CONFIG.PEDESTRIAN_TYPES:
                node_groups[node.type]['sizes'].append(CONFIG.NODE_SIZE_PEDESTRIAN)
            elif node.type in CONFIG.CONNECTION_TYPES:
                node_groups[node.type]['sizes'].append(CONFIG.NODE_SIZE_CONNECTION)
            elif node.type in CONFIG.VERTICAL_TYPES:
                node_groups[node.type]['sizes'].append(CONFIG.NODE_SIZE_VERTICAL)
            elif node.type in CONFIG.ROOM_TYPES:
                node_groups[node.type]['sizes'].append(CONFIG.NODE_SIZE_ROOM)
            elif node.type in CONFIG.OUTSIDE_TYPES:
                node_groups[node.type]['sizes'].append(CONFIG.NODE_SIZE_OUTSIDE)
            else:
                node_groups[node.type]['sizes'].append(CONFIG.NODE_SIZE_DEFAULT)
        
        # 分类连接线（水平和垂直）
        horizontal_edges = {'x': [], 'y': [], 'z': []}
        vertical_edges = {'x': [], 'y': [], 'z': []}
        
        for edge in self.super_graph.edges():
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            
            # 判断是否为垂直连接线（通过检查z坐标是否相同）
            if abs(z0 - z1) > 0.01:  # 允许一点误差
                if CONFIG.IMAGE_MIRROR:
                    vertical_edges['x'].extend([self.width - x0, self.width - x1, None])
                else:
                    vertical_edges['x'].extend([x0, x1, None])
                vertical_edges['y'].extend([y0, y1, None])
                vertical_edges['z'].extend([z0, z1, None])
            else:
                if CONFIG.IMAGE_MIRROR:
                    horizontal_edges['x'].extend([self.width - x0, self.width - x1, None])
                else:
                    horizontal_edges['x'].extend([x0, x1, None])
                horizontal_edges['y'].extend([y0, y1, None])
                horizontal_edges['z'].extend([z0, z1, None])
        
        # 创建水平连接线轨迹
        horizontal_edge_trace = go.Scatter3d(
            x=horizontal_edges['x'],
            y=horizontal_edges['y'],
            z=horizontal_edges['z'],
            line=dict(width=CONFIG.EDGE_WIDTH, color=CONFIG.HORIZONTAL_EDGE_COLOR),
            hoverinfo='none',
            mode='lines',
            name='水平连接'  # 添加图例名称
        )
        
        # 创建垂直连接线轨迹
        vertical_edge_trace = go.Scatter3d(
            x=vertical_edges['x'],
            y=vertical_edges['y'],
            z=vertical_edges['z'],
            line=dict(width=CONFIG.EDGE_WIDTH, color=CONFIG.VERTICAL_EDGE_COLOR),
            hoverinfo='none',
            mode='lines',
            name='垂直连接'  # 添加图例名称
        )
        
        # 为每种节点类型创建轨迹
        node_traces = []
        for node_type, data in node_groups.items():
            if not data['x']:  # 跳过空分组
                continue
                
            node_trace = go.Scatter3d(
                x=data['x'],
                y=data['y'],
                z=data['z'],
                mode='markers+text',
                hoverinfo='text',
                name=node_type,  # 添加图例名称
                marker=dict(
                    size=data['sizes'],
                    sizemode='diameter',  # 确保缩放时节点大小保持不变
                    opacity=CONFIG.NODE_OPACITY,
                    color=self.get_color_for_type(node_type),  # 使用类型对应的颜色
                    line_width=2
                ),
                text=data['text'],
                textposition="top center"
            )
            node_traces.append(node_trace)
        
        # 获取楼层选择控件
        floor_controls = self._create_floor_selection_controls(all_z_levels, min_z, max_z)
        
        # 创建图形
        fig = go.Figure(
            data=[horizontal_edge_trace, vertical_edge_trace] + node_traces,
            layout=go.Layout(
                title='3D Network Graph',
                showlegend=True,  # 显示图例
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                legend=dict(
                    itemsizing='constant',  # 图例项大小一致
                    font=dict(size=10),  # 图例文字大小
                ),
                scene=dict(
                    xaxis=dict(title='X'),
                    yaxis=dict(title='Y'),
                    zaxis=dict(
                        title='Z',
                        range=[min_z, max_z]  # 初始Z轴范围
                    ),
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.5)  # 设置初始视角
                    )
                ),
                # 添加楼层选择控件
                sliders=floor_controls["sliders"],
            )
        )

        if save:
            fig.write_html(CONFIG.RESULT_PATH / '3D_network_plotly.html')  # 保存为 HTML
        else:
            fig.show()
            
        return fig
        
    def get_color_for_type(self, node_type):
        """根据节点类型获取对应的颜色"""
        if not COLOR_MAP:
            return '#1f77b4'
        try:
            # 尝试从COLOR_MAP获取颜色
            for color_tuple, info in COLOR_MAP.items():
                if info['name'] == node_type:
                    # 将RGB元组转为十六进制颜色
                    return f'rgb{color_tuple}'
        except:
            pass
        # 默认颜色
        return '#1f77b4'

def calculate_room_travel_times(graph: nx.Graph):
    """
    计算房间之间的通行时间
    """
    room_nodes = [node for node in graph.nodes if node.type in CONFIG.ROOM_TYPES]

    room_graph = graph # 使用完整的图进行路径查找

    room_type_times ={}

    # 定义权重函数：边的权重等于目标节点的通行时间
    weight_func = lambda u, v, data: v.time

    for start_node in room_nodes:
        room_type1 = start_node.type

        # 使用Dijkstra计算从start_node到所有其他节点的最短路径长度（基于时间）
        # lengths 字典包含 target_node -> path_length 的映射
        # path_length 是从 start_node 到 target_node 的路径上所有节点的时间总和（不包括 start_node.time）
        try:
            lengths = nx.single_source_dijkstra_path_length(room_graph, start_node, weight=weight_func)
        except nx.NetworkXNoPath:
            # 如果 start_node 无法到达任何其他节点（例如图不连通），则跳过
            continue

        if room_type1 not in room_type_times:
            room_type_times[room_type1] = {}

        for target_node in room_nodes:
            room_type2 = target_node.type

            if target_node == start_node:
                # 到自身的通行时间就是节点自身的通行时间
                room_type_times[room_type1][room_type2] = start_node.time
                continue

            # 检查 target_node 是否可达
            if target_node in lengths:
                # 总时间 = Dijkstra路径长度 (不含起点) + 起点时间
                total_time = lengths[target_node] + start_node.time
                room_type_times[room_type1][room_type2] = total_time
            else:
                # 如果 target_node 不可达，设置时间为无穷大
                room_type_times[room_type1][room_type2] = np.inf

    # 保存为CSV文件
    os.makedirs(CONFIG.RESULT_PATH, exist_ok=True)

    # 获取所有唯一的房间类型
    all_room_types = sorted(list(set(node.type for node in room_nodes)))

    with open(CONFIG.RESULT_PATH / 'result.csv', 'w', newline='', encoding='utf-8') as f:
        import csv
        writer = csv.writer(f)

        # 写入表头
        header = ['房间类型'] + all_room_types
        writer.writerow(header)

        # 写入数据行
        for start_type in all_room_types:
            row = [start_type]
            for end_type in all_room_types:
                if start_type in room_type_times and end_type in room_type_times[start_type]:
                    value = room_type_times[start_type][end_type]
                    # 如果是无穷大，写入特殊标记
                    if value == np.inf:
                        row.append('∞')
                    else:
                        row.append(value)
                else:
                    # 如果没有计算出时间（例如某个类型没有节点），则标记为N/A
                    row.append('N/A')
            writer.writerow(row)

    return room_type_times