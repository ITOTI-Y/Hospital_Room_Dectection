"""
Defines plotter classes for visualizing network graphs using Matplotlib and Plotly.
"""
import abc
import pathlib
from src.rl_optimizer.utils.setup import setup_logger
import networkx as nx
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Tuple, Any, List, Optional

from src.config import NetworkConfig  # 依赖配置类
from src.network.node import Node     # 依赖节点类

logger = setup_logger(__name__)


class BasePlotter(abc.ABC):
    """
    Abstract base class for graph plotters.
    """

    def __init__(self,
                 config: NetworkConfig,
                 color_map_data: Dict[Tuple[int, int, int], Dict[str, Any]]):
        """
        Initializes the BasePlotter.

        Args:
            config: The network configuration object.
            color_map_data: The global color map dictionary.
        """
        self.config = config
        self.color_map_data = color_map_data
        self.type_to_plot_color_cache: Dict[str, str] = {}  # 缓存节点类型到绘图颜色的映射

    def _validate_node_coordinates(self, node: Node, node_id: str = None) -> bool:
        """
        验证节点坐标的有效性。
        
        Args:
            node: 要验证的节点对象
            node_id: 节点ID（用于错误报告）
            
        Returns:
            bool: 坐标是否有效
        """
        try:
            if not hasattr(node, 'x') or not hasattr(node, 'y') or not hasattr(node, 'z'):
                return False
            
            coords = [node.x, node.y, node.z]
            return all(
                isinstance(c, (int, float)) and 
                not np.isnan(c) and 
                np.isfinite(c) 
                for c in coords
            )
        except (TypeError, AttributeError):
            return False

    def _get_edge_type_config(self) -> Dict[str, Dict[str, Any]]:
        """
        获取边类型的样式配置。
        
        Returns:
            Dict: 边类型配置字典
        """
        return {
            'horizontal': {
                'color': self.config.HORIZONTAL_EDGE_COLOR,
                'width': self.config.EDGE_WIDTH,
                'name': 'Horizontal Connection',
                'dash': None
            },
            'vertical': {
                'color': self.config.VERTICAL_EDGE_COLOR, 
                'width': self.config.EDGE_WIDTH * 1.5,
                'name': 'Vertical Connection',
                'dash': None
            },
            'door': {
                'color': self.config.DOOR_EDGE_COLOR,
                'width': self.config.EDGE_WIDTH * 1.2,
                'name': 'Door Connection',
                'dash': None
            },
            'special': {
                'color': self.config.SPECIAL_EDGE_COLOR,
                'width': self.config.EDGE_WIDTH * 1.3,
                'name': 'Special Connection',
                'dash': None
            }
        }

    def _classify_edge_type(self, start_node: Node, end_node: Node, edge_attr: dict, z0: float, z1: float) -> str:
        """
        智能分类边的类型，用于不同的可视化样式。
        
        使用精确匹配而非包含匹配，避免误分类问题。
        
        Args:
            start_node: 起始节点对象
            end_node: 终止节点对象
            edge_attr: 边的属性字典
            z0: 起始节点Z坐标
            z1: 终止节点Z坐标
            
        Returns:
            边类型字符串：'horizontal', 'vertical', 'door', 'special'
            
        Raises:
            ValueError: 当节点对象无效时抛出异常
        """
        # 输入验证
        if not isinstance(start_node, Node) or not isinstance(end_node, Node):
            raise ValueError(f"Invalid node objects: start_node={type(start_node)}, end_node={type(end_node)}")
        
        if not hasattr(start_node, 'node_type') or not hasattr(end_node, 'node_type'):
            raise ValueError("Node objects must have 'node_type' attribute")
        
        if start_node.node_type is None or end_node.node_type is None:
            raise ValueError("Node types cannot be None")
        
        # 检查Z坐标有效性
        if not isinstance(z0, (int, float)) or not isinstance(z1, (int, float)):
            raise ValueError(f"Invalid Z coordinates: z0={type(z0)}, z1={type(z1)}")
        
        # 检查Z坐标数值有效性（NaN和无穷大）
        if not (np.isfinite(z0) and np.isfinite(z1)):
            raise ValueError(f"Z coordinates must be finite: z0={z0}, z1={z1}")
        
        # 检查是否为垂直连接（跨楼层）
        z_diff_threshold = getattr(self.config, 'VERTICAL_CONNECTION_Z_THRESHOLD', 0.1)
        if abs(z0 - z1) > z_diff_threshold:
            return 'vertical'
        
        # 使用配置中的类型定义进行精确匹配
        start_type = str(start_node.node_type).strip()
        end_type = str(end_node.node_type).strip()

        # 检查是否涉及Door连接（使用配置中的CONNECTION_TYPES）
        connection_types = set(getattr(self.config, 'CONNECTION_TYPES', ['Door']))
        if start_type in connection_types or end_type in connection_types:
            return 'door'
        
        # 检查是否为特殊类型连接（使用配置中的VERTICAL_TYPES）
        vertical_types = set(getattr(self.config, 'VERTICAL_TYPES', ['电梯', '扶梯', '楼梯']))
        if start_type in vertical_types or end_type in vertical_types:
            return 'special'
        
        # 默认为水平连接
        return 'horizontal'

    def _get_node_color(self, node_type: str) -> str:
        """
        Determines the plotting color for a given node type.

        Uses colors from `color_map_data` if `NODE_COLOR_FROM_MAP` is True in config,
        otherwise uses a default Plotly color. Caches results.

        Args:
            node_type: The type of the node (e.g., 'Room', 'Door').

        Returns:
            A string representing the color (e.g., 'rgb(R,G,B)' or a named Plotly color).
        """
        if node_type in self.type_to_plot_color_cache:
            return self.type_to_plot_color_cache[node_type]

        default_plotly_color = '#1f77b4'  # Plotly's default blue

        if self.config.NODE_COLOR_FROM_MAP and self.color_map_data:
            for rgb_tuple, details in self.color_map_data.items():
                if details.get('name') == node_type:
                    color_str = f'rgb{rgb_tuple}'
                    self.type_to_plot_color_cache[node_type] = color_str
                    return color_str

        self.type_to_plot_color_cache[node_type] = default_plotly_color
        return default_plotly_color

    @abc.abstractmethod
    def plot(self,
             graph: nx.Graph,
             output_path: Optional[pathlib.Path] = None,
             title: str = "Network Graph",
             # For Plotly layout, original image width
             graph_width: Optional[int] = None,
             # For Plotly layout, original image height
             graph_height: Optional[int] = None,
             # For SuperNetwork floor labels
             floor_z_map: Optional[Dict[int, float]] = None
             ):
        """
        Abstract method to plot the graph.

        Args:
            graph: The NetworkX graph to plot.
            output_path: Optional path to save the plot. If None, displays the plot.
            title: The title for the plot.
            graph_width: Original width of the (floor plan) image space. Used by Plotly.
            graph_height: Original height of the (floor plan) image space. Used by Plotly.
            floor_z_map: Mapping from floor number to Z-coordinate, for floor slider labels.
        """
        pass


class PlotlyPlotter(BasePlotter):
    """
    Generates interactive 3D network graph visualizations using Plotly.
    """

    def _create_floor_selection_controls(self,
                                         all_z_levels: List[float],
                                         min_z: float, max_z: float,
                                         floor_z_map_for_labels: Optional[Dict[int,
                                                                               float]] = None,
                                         base_floor_for_labels: int = 0
                                         ) -> Dict[str, Any]:
        """
        Creates slider controls for selecting and viewing individual floors or all floors.
        Args:
            all_z_levels: Sorted list of unique Z-coordinates present in the graph.
            min_z: Minimum Z-coordinate.
            max_z: Maximum Z-coordinate.
            floor_z_map_for_labels: Mapping from actual floor number to Z-coordinate.
            base_floor_for_labels: The base floor number for labeling (e.g. 0 for ground, 1 for first).
        """
        if not all_z_levels:
            return {"sliders": []}

        # Create floor labels. Try to map Z-levels back to "human-readable" floor numbers.
        z_to_floor_label_map: Dict[float, str] = {}
        if floor_z_map_for_labels:
            # Invert floor_z_map_for_labels to map z -> floor_num for easier lookup
            # Handle potential multiple floors at the same Z (unlikely with good input)
            z_to_floor_num: Dict[float, List[int]] = {}
            for fn, z_val in floor_z_map_for_labels.items():
                z_to_floor_num.setdefault(z_val, []).append(fn)

            for z_level in all_z_levels:
                floor_nums_at_z = z_to_floor_num.get(z_level)
                if floor_nums_at_z:
                    # If multiple floor numbers map to the same z_level, list them or take first
                    f_num_str = "/".join(map(str, sorted(floor_nums_at_z)))
                    # e.g., F1, F-1/B1
                    z_to_floor_label_map[z_level] = f"F{f_num_str}"
                # Fallback if z_level not in map (should not happen if map is complete)
                else:
                    z_to_floor_label_map[z_level] = f"Z={z_level:.1f}"
        else:  # Fallback if no floor_z_map is provided
            for i, z_level in enumerate(all_z_levels):
                # Attempt simple labeling if base_floor is known
                floor_num_guess = base_floor_for_labels + i  # This is a rough guess
                z_to_floor_label_map[z_level] = f"F{floor_num_guess} (Z={z_level:.1f})"

        slider_steps = []
        for z_level in all_z_levels:
            label = z_to_floor_label_map.get(z_level, f"Z={z_level:.1f}")
            slider_steps.append(dict(
                label=label,
                method="relayout",
                args=[{"scene.zaxis.range": [z_level - self.config.DEFAULT_FLOOR_HEIGHT / 2 + 0.1,
                                             z_level + self.config.DEFAULT_FLOOR_HEIGHT / 2 - 0.1]}]  # View single floor
            ))

        # Add a step to show all floors
        slider_steps.append(dict(
            label="All Floors",
            method="relayout",
            args=[{"scene.zaxis.range": [min_z - self.config.DEFAULT_FLOOR_HEIGHT * 0.5,
                                         max_z + self.config.DEFAULT_FLOOR_HEIGHT * 0.5]}]  # View all
        ))

        sliders = [dict(
            active=len(all_z_levels),  # Default to "All Floors"
            currentvalue={"prefix": "Current Display: "},
            pad={"t": 50},
            steps=slider_steps,
            name="Floor Selection"
        )]
        return {"sliders": sliders}

    def plot(self,
             graph: nx.Graph,
             output_path: Optional[pathlib.Path] = None,
             title: str = "3D Network Graph",
             graph_width: Optional[int] = None,
             graph_height: Optional[int] = None,
             floor_z_map: Optional[Dict[int, float]] = None
             ):
        if not graph.nodes:
            # Ensure logger is defined/imported
            logger.warning("PlotlyPlotter: Graph has no nodes to plot.")
            return

        node_traces = []
        edge_traces = []  # Renamed from edge_trace to edge_traces as it's a list

        nodes_data_by_type: Dict[str, Dict[str, list]] = {}
        all_node_objects = [data.get('node_obj', node_id)
                            for node_id, data in graph.nodes(data=True)]
        all_node_objects = [n for n in all_node_objects if isinstance(n, Node)]

        if not all_node_objects:
            logger.warning(
                "PlotlyPlotter: No Node objects found in graph nodes. Cannot plot.")
            return

        all_z_coords_present = sorted(
            list(set(n.z for n in all_node_objects)))
        min_z = min(all_z_coords_present) if all_z_coords_present else 0
        max_z = max(all_z_coords_present) if all_z_coords_present else 0

        for node_obj in all_node_objects:
            node_type = node_obj.node_type
            e_name = node_obj.e_name
            code = node_obj.code
            if node_type not in nodes_data_by_type:
                nodes_data_by_type[node_type] = {
                    'x': [], 'y': [], 'z': [],
                    'visible_text': [],  # For text always visible next to node
                    'hover_text': [],   # For text visible on hover
                    'sizes': [],
                    'ids': []
                }

            x, y, z = node_obj.x, node_obj.y, node_obj.z
            plot_x = (
                graph_width - x) if self.config.IMAGE_MIRROR and graph_width is not None else x

            nodes_data_by_type[node_type]['x'].append(plot_x)
            nodes_data_by_type[node_type]['y'].append(y)
            nodes_data_by_type[node_type]['z'].append(z)
            nodes_data_by_type[node_type]['ids'].append(node_obj.id)

            # --- Text Configuration ---
            # 1. Visible text (always shown next to the marker if mode includes 'text')
            #    Only show node_type if SHOW_PEDESTRIAN_LABELS is True or it's not a pedestrian node.
            #    Otherwise, show empty string to hide permanent text for certain types.
            is_ped_type = node_type in self.config.PEDESTRIAN_TYPES
            can_show_permanent_label = not is_ped_type or self.config.SHOW_PEDESTRIAN_LABELS

            nodes_data_by_type[node_type]['visible_text'].append(
                code if can_show_permanent_label else "")

            # 2. Hover text (always detailed)
            hover_label = (
                f"ID: {node_obj.id}<br>"
                f"Type: {node_type}<br>"
                f"Pos: ({x},{y},{z})<br>"
                f"Time: {node_obj.time:.2f}<br>"
                f"Area: {getattr(node_obj, 'area', 0):.2f}"
            )
            if hasattr(node_obj, 'door_type') and node_obj.door_type:
                hover_label += f"<br>Door: {node_obj.door_type}"
            nodes_data_by_type[node_type]['hover_text'].append(hover_label)

            # Node size
            size = self.config.NODE_SIZE_DEFAULT
            if is_ped_type:
                size = self.config.NODE_SIZE_PEDESTRIAN
            elif node_type in self.config.CONNECTION_TYPES:
                size = self.config.NODE_SIZE_CONNECTION
            elif node_type in self.config.VERTICAL_TYPES:
                size = self.config.NODE_SIZE_VERTICAL
            elif node_type in self.config.ROOM_TYPES:
                size = self.config.NODE_SIZE_ROOM
            elif node_type in self.config.OUTSIDE_TYPES:
                size = self.config.NODE_SIZE_OUTSIDE
            nodes_data_by_type[node_type]['sizes'].append(size)

        for node_type, data in nodes_data_by_type.items():
            if not data['x']:
                continue

            # Determine mode: if all 'visible_text' for this type are empty, just use 'markers'
            # Otherwise, use 'markers+text' to show the type.
            current_mode = 'markers'
            # Check if any visible text is non-empty
            if any(vt for vt in data['visible_text']):
                current_mode = 'markers+text'

            # If SHOW_PEDESTRIAN_LABELS is False and it's a pedestrian type, override to 'markers'
            if node_type in self.config.PEDESTRIAN_TYPES and not self.config.SHOW_PEDESTRIAN_LABELS:
                current_mode = 'markers'

            node_trace = go.Scatter3d(
                x=data['x'], y=data['y'], z=data['z'],
                mode=current_mode,  # Dynamically set mode
                marker=dict(
                    size=data['sizes'],
                    sizemode='diameter',  # This should make size in screen pixels
                    color=self._get_node_color(node_type),
                    opacity=self.config.NODE_OPACITY,
                    line=dict(width=1, color='DarkSlateGrey')
                ),
                # Text to display next to markers if mode includes 'text'
                text=data['visible_text'],
                hovertext=data['hover_text'],  # Text for hover box
                # Use 'text' from hovertext (Plotly default is 'all')
                hoverinfo='text',
                # if hovertext is set, hoverinfo='text' uses hovertext.
                # if hovertext is not set, hoverinfo='text' uses the 'text' property.
                name=node_type,
                customdata=data['ids'],
                textposition="top center",
                textfont=dict(  # Optional: style the permanently visible text
                    size=9,  # Smaller font for permanent labels
                    # color='black'
                )
            )
            node_traces.append(node_trace)

        # --- Prepare Edge Data with Enhanced Classification ---
        # 存储不同类型边的坐标数据
        edge_data = {
            'horizontal': {'x': [], 'y': [], 'z': []},
            'vertical': {'x': [], 'y': [], 'z': []},
            'door': {'x': [], 'y': [], 'z': []},
            'special': {'x': [], 'y': [], 'z': []}
        }
        
        # 性能和错误统计
        edges_processed = 0
        edges_skipped = 0
        classification_errors = 0
        coordinate_errors = 0
        
        # 输入验证
        if not graph or len(graph.edges) == 0:
            logger.warning("PlotlyPlotter: 图中没有边需要处理")
            edge_traces = []
        else:
            # 创建节点ID到Node对象的映射以提高查找性能
            node_id_to_obj = {}
            invalid_node_count = 0
            
            for node_id, node_data in graph.nodes(data=True):
                node_obj = node_data.get('node_obj')
                if isinstance(node_obj, Node):
                    node_id_to_obj[node_id] = node_obj
                else:
                    invalid_node_count += 1
            
            if invalid_node_count > 0:
                logger.warning(f"发现 {invalid_node_count} 个无效节点对象")
            
            logger.debug(f"创建节点映射表完成，包含 {len(node_id_to_obj)} 个有效节点")

            # 遍历所有边并正确获取节点对象
            for edge_start_id, edge_end_id, edge_attr in graph.edges(data=True):
                edges_processed += 1
                
                try:
                    # 边界条件检查：边属性
                    if edge_attr is None:
                        edge_attr = {}
                    
                    # 通过节点ID获取Node对象
                    start_node = node_id_to_obj.get(edge_start_id)
                    end_node = node_id_to_obj.get(edge_end_id)
                    
                    # 验证节点对象存在性和有效性
                    if not isinstance(start_node, Node) or not isinstance(end_node, Node):
                        edges_skipped += 1
                        if edges_processed <= 10:  # 只记录前10个错误避免日志泛滥
                            logger.warning(f"边 ({edge_start_id}, {edge_end_id}) 的节点对象无效: "
                                         f"start_node={type(start_node)}, end_node={type(end_node)}")
                        continue
                    
                    # 验证节点坐标有效性
                    if not self._validate_node_coordinates(start_node, str(edge_start_id)) or \
                       not self._validate_node_coordinates(end_node, str(edge_end_id)):
                        coordinate_errors += 1
                        edges_skipped += 1
                        if coordinate_errors <= 5:
                            logger.warning(f"边 ({edge_start_id}, {edge_end_id}) 的节点坐标无效")
                        continue
                    
                    # 获取节点位置信息
                    x0, y0, z0 = start_node.x, start_node.y, start_node.z
                    x1, y1, z1 = end_node.x, end_node.y, end_node.z
                    
                    # 应用镜像变换（如果启用）
                    if self.config.IMAGE_MIRROR and graph_width is not None and isinstance(graph_width, (int, float)):
                        plot_x0 = graph_width - x0
                        plot_x1 = graph_width - x1
                    else:
                        plot_x0, plot_x1 = x0, x1
                    
                    # 智能边类型分类（带异常处理）
                    try:
                        edge_type = self._classify_edge_type(start_node, end_node, edge_attr, z0, z1)
                        
                        # 验证分类结果
                        if edge_type not in edge_data:
                            classification_errors += 1
                            if classification_errors <= 5:
                                logger.warning(f"未知的边类型 '{edge_type}'，使用默认类型 'horizontal'")
                            edge_type = 'horizontal'
                        
                    except Exception as class_e:
                        classification_errors += 1
                        edges_skipped += 1
                        if classification_errors <= 5:
                            logger.warning(f"分类边 ({edge_start_id}, {edge_end_id}) 时出错: {class_e}，跳过该边")
                        continue
                    
                    # 将边数据添加到相应类别
                    edge_data[edge_type]['x'].extend([plot_x0, plot_x1, None])
                    edge_data[edge_type]['y'].extend([y0, y1, None])
                    edge_data[edge_type]['z'].extend([z0, z1, None])
                    
                except Exception as e:
                    edges_skipped += 1
                    logger.warning(f"处理边 ({edge_start_id}, {edge_end_id}) 时发生未知错误: {type(e).__name__}: {str(e)}")
                    continue
            
            # 详细统计报告
            successful_edges = edges_processed - edges_skipped
            logger.info(f"边处理统计: 总计 {edges_processed} 条边，成功处理 {successful_edges} 条，跳过 {edges_skipped} 条")
            if coordinate_errors > 0:
                logger.info(f"坐标错误: {coordinate_errors} 条边")
            if classification_errors > 0:
                logger.info(f"分类错误: {classification_errors} 条边")
            
            # 按类型统计边数
            type_counts = {edge_type: len(data['x']) // 3 for edge_type, data in edge_data.items() if data['x']}
            if type_counts:
                logger.info(f"边类型分布: {type_counts}")

            # 创建不同类型边的可视化轨迹
            edge_traces = []
            edge_style_config = self._get_edge_type_config()
        
        # 为每种边类型创建Scatter3d轨迹
        for edge_type, data in edge_data.items():
            if not data['x']:  # 跳过空数据
                continue
                
            style = edge_style_config[edge_type]
            line_config = {
                'color': style['color'],
                'width': style['width']
            }
            if style['dash']:
                line_config['dash'] = style['dash']
            
            edge_trace = go.Scatter3d(
                x=data['x'],
                y=data['y'], 
                z=data['z'],
                mode='lines',
                line=line_config,
                hoverinfo='none',
                name=style['name'],
                showlegend=True,  # 显示在图例中
                legendgroup=f'edges_{edge_type}'  # 分组管理
            )
            edge_traces.append(edge_trace)
            
        logger.info(f"创建了 {len(edge_traces)} 种类型的边轨迹")

        # --- Layout and Figure (remains largely the same) ---
        layout = go.Layout(
            title=title,
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            scene=dict(
                xaxis=dict(
                    title='X', autorange='reversed' if self.config.IMAGE_MIRROR else True),
                yaxis=dict(
                    title='Y',
                    autorange='reversed', # 反转Y轴
                ),
                zaxis=dict(title='Z (Floor)'),
                aspectmode="manual",
                aspectratio=dict(
                    x=self.config.X_AXIS_RATIO,
                    y=self.config.Y_AXIS_RATIO,
                    z=self.config.Z_AXIS_RATIO
                ),
                camera=dict(
                    # projection=dict(type='orthographic'),
                    eye=dict(x=1.25, y=1.25, z=1.25)
                    )
            ),
            legend=dict(
                orientation="v",    # 垂直排列
                x=0.02,             # X 位置 (靠近左边缘)
                y=1.0,              # Y 位置 (靠近顶部)
                xanchor="left",     # X 锚点
                yanchor="top",      # Y 锚点
                bgcolor="rgba(255, 255, 255, 0.7)", # 可选：浅色背景提高可读性
                # bordercolor="rgba(120, 120, 120, 0.7)", # 可选：边框颜色
                # borderwidth=1         # 可选：边框宽度
            )
        )

        if len(all_z_coords_present) > 1:
            floor_controls = self._create_floor_selection_controls(
                all_z_coords_present, min_z, max_z, floor_z_map)
            layout.update(floor_controls)

        fig = go.Figure(data=node_traces + edge_traces, layout=layout)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(output_path), config=self.config.PLOTLY_CONFIG)
            # Ensure logger
            logger.info(f"Plotly graph saved to {output_path}")
        else:
            fig.show()
