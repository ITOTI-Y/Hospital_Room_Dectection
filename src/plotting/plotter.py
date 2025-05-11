"""
Defines plotter classes for visualizing network graphs using Matplotlib and Plotly.
"""
import abc
import pathlib
import logging
import networkx as nx
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Tuple, Any, List, Optional

from src.config import NetworkConfig  # 依赖配置类
from src.graph.node import Node     # 依赖节点类

logger = logging.getLogger(__name__)


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
            label="所有楼层",
            method="relayout",
            args=[{"scene.zaxis.range": [min_z - self.config.DEFAULT_FLOOR_HEIGHT * 0.5,
                                         max_z + self.config.DEFAULT_FLOOR_HEIGHT * 0.5]}]  # View all
        ))

        sliders = [dict(
            active=len(all_z_levels),  # Default to "All Floors"
            currentvalue={"prefix": "当前显示: "},
            pad={"t": 50},
            steps=slider_steps,
            name="楼层选择"
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
            list(set(n.pos[2] for n in all_node_objects)))
        min_z = min(all_z_coords_present) if all_z_coords_present else 0
        max_z = max(all_z_coords_present) if all_z_coords_present else 0

        for node_obj in all_node_objects:
            node_type = node_obj.node_type
            if node_type not in nodes_data_by_type:
                nodes_data_by_type[node_type] = {
                    'x': [], 'y': [], 'z': [],
                    'visible_text': [],  # For text always visible next to node
                    'hover_text': [],   # For text visible on hover
                    'sizes': [],
                    'ids': []
                }

            x, y, z = node_obj.pos
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
                node_type if can_show_permanent_label else "")

            # 2. Hover text (always detailed)
            hover_label = (
                f"ID: {node_obj.id}<br>"
                f"Type: {node_type}<br>"
                f"Pos: ({x},{y},{z})<br>"
                f"Time: {node_obj.time:.2f}<br>"
                f"Area: {node_obj.area:.2f}"
            )
            if node_obj.door_type:
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

        # --- Prepare Edge Data (remains largely the same) ---
        edge_x_horiz, edge_y_horiz, edge_z_horiz = [], [], []
        edge_x_vert, edge_y_vert, edge_z_vert = [], [], []

        for edge_start_node, edge_end_node in graph.edges():
            if not (isinstance(edge_start_node, Node) and isinstance(edge_end_node, Node)):
                continue

            x0, y0, z0 = edge_start_node.pos
            x1, y1, z1 = edge_end_node.pos
            plot_x0 = (
                graph_width - x0) if self.config.IMAGE_MIRROR and graph_width is not None else x0
            plot_x1 = (
                graph_width - x1) if self.config.IMAGE_MIRROR and graph_width is not None else x1

            if abs(z0 - z1) < 0.1:
                edge_x_horiz.extend([plot_x0, plot_x1, None])
                edge_y_horiz.extend([y0, y1, None])
                edge_z_horiz.extend([z0, z1, None])
            else:
                edge_x_vert.extend([plot_x0, plot_x1, None])
                edge_y_vert.extend([y0, y1, None])
                edge_z_vert.extend([z0, z1, None])

        if edge_x_horiz:
            edge_traces.append(go.Scatter3d(
                x=edge_x_horiz, y=edge_y_horiz, z=edge_z_horiz,
                mode='lines',
                line=dict(color=self.config.HORIZONTAL_EDGE_COLOR,
                          width=self.config.EDGE_WIDTH),
                hoverinfo='none', name='水平连接'
            ))
        if edge_x_vert:
            edge_traces.append(go.Scatter3d(
                x=edge_x_vert, y=edge_y_vert, z=edge_z_vert,
                mode='lines',
                line=dict(color=self.config.VERTICAL_EDGE_COLOR,
                          width=self.config.EDGE_WIDTH),
                hoverinfo='none', name='垂直连接'
            ))

        # --- Layout and Figure (remains largely the same) ---
        layout = go.Layout(
            title=title,
            showlegend=True,
            hovermode='closest',  # Important for hover behavior
            margin=dict(b=20, l=5, r=5, t=40),
            scene=dict(
                xaxis=dict(
                    title='X', autorange='reversed' if self.config.IMAGE_MIRROR else True),
                yaxis=dict(
                    title='Y',
                    autorange='reversed', # 反转Y轴
                ),
                zaxis=dict(title='Z (楼层)', range=[min_z - 1, max_z + 1]),
                aspectmode='data',  # 'data' is often good for spatial data
                camera=dict(eye=dict(x=1.25, y=1.25, z=1.25))
            ),
            legend=dict(
                orientation="v",    # 垂直排列
                x=0.02,             # X 位置 (靠近左边缘)
                y=1.0,              # Y 位置 (靠近顶部)
                xanchor="left",     # X 锚点
                yanchor="top",      # Y 锚点
                bgcolor="rgba(255, 255, 255, 0.7)", # 可选：浅色背景提高可读性
                bordercolor="rgba(120, 120, 120, 0.7)", # 可选：边框颜色
                borderwidth=1         # 可选：边框宽度
            )
        )

        if len(all_z_coords_present) > 1:
            floor_controls = self._create_floor_selection_controls(
                all_z_coords_present, min_z, max_z, floor_z_map)
            layout.update(floor_controls)

        fig = go.Figure(data=node_traces + edge_traces, layout=layout)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(output_path))
            # Ensure logger
            logger.info(f"Plotly graph saved to {output_path}")
        else:
            fig.show()
