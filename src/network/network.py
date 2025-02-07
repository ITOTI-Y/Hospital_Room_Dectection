import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import cv2
import plotly.graph_objects as go
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
        self.graph = nx.Graph()
        self.id_to_node = {}

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
        self.image = np.asarray(Image.open(image_path))
        self.id_map = np.zeros_like(self.image, dtype=int)
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]
        self._create_outside_mask()
        self._create_room_nodes(zlevel=zlevel)
        self._create_pedestrian_nodes(zlevel=zlevel)
        if outside:
            self._create_outside_nodes(zlevel=zlevel)
        self._create_connection_nodes(zlevel=zlevel)
        self._connect_pedestrian_connection(outside=outside)
        return self.graph

    # def plot(self, save: bool = False):
    #     """
    #     可视化节点图
    #     """
    #     # 将图片尺寸从像素转换为英寸（考虑 DPI）
    #     dpi = 100
    #     width_inches = self.width / dpi
    #     height_inches = self.height / dpi

    #     # 创建指定尺寸的图形
    #     fig = plt.figure(figsize=(width_inches, height_inches), dpi=dpi)
    #     # 移除图形周围的空白边距
    #     ax = plt.Axes(fig, [0., 0., 1., 1.])
    #     ax.set_axis_off()
    #     fig.add_axes(ax)

    #     # 显示图像和节点
    #     ax.imshow(self.image)
    #     pos = {node:node.pos for node in self.graph.nodes}
    #     labels = {node:node.type for node in self.graph.nodes}
    #     # 设置中文字体
    #     plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
    #     plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
    #     nx.draw(self.graph, pos, labels=labels, with_labels=True,
    #             node_size=50, node_color='red', font_size=8,
    #             font_weight='bold', ax=ax)

    #     if save:
    #         plt.savefig('./debug/network.png',
    #                     dpi=dpi,
    #                     bbox_inches='tight',
    #                     pad_inches=0)
    #         plt.close()
    #     else:
    #         plt.show()
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

        # 绘制节点
        ax.scatter(x, y, z, c='red', s=50)  # 使用 scatter 绘制散点图

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
        edge_x = []
        edge_y = []
        edge_z = []
        for edge in self.graph.edges():
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])  # None 用于分隔线段
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])

        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_x = []
        node_y = []
        node_z = []
        node_text = []
        for node in self.graph.nodes():
            x, y, z = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            node_text.append(f"{node.type}")

        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers+text',  # 同时显示标记和文本
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                reversescale=True,
                color=[],
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',  # 只需要设置 title
                    xanchor='left',
                ),
                line_width=2),
            text=node_text,       # 设置节点文本
            textposition="top center")  # 文本位置

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
            title='Network Graph',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            scene=dict(  # 设置3D场景
                xaxis=dict(title='X'),
                yaxis=dict(title='Y'),
                zaxis=dict(title='Z'),
            )
        ))

        if save:
            fig.write_html('./debug/network_plotly.html')  # 保存为 HTML
        else:
            fig.show()
