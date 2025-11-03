"""
Defines plotter classes for visualizing network graphs using Matplotlib and Plotly.
"""

import abc
import pathlib
from typing import Any

import networkx as nx
import numpy as np
import plotly.graph_objects as go
from loguru import logger

from src.config import graph_config

logger = logger.bind(module=__name__)


class BasePlotter(abc.ABC):
    """
    Abstract base class for graph plotters.
    """

    def __init__(self):
        """
        Initializes the BasePlotter.
        """
        self.plotter_config = graph_config.get_plotter_config()
        self.node_defs = graph_config.get_node_definitions()
        self.super_network_config = graph_config.get_super_network_config()

    def _validate_node_coordinates(
        self, node_data: dict[str, Any], node_id: Any
    ) -> bool:
        """
        Validates node coordinates from the graph attributes.
        """
        try:
            coords = [
                node_data.get("pos_x"),
                node_data.get("pos_y"),
                node_data.get("pos_z"),
            ]
            return all(isinstance(c, (int, float)) and np.isfinite(c) for c in coords)
        except (TypeError, AttributeError):
            return False

    def _get_edge_style_config(self) -> dict[str, dict[str, Any]]:
        """
        Gets edge style configuration from the plotter config.
        """
        return self.plotter_config.get("edge_styles", {})

    def _classify_edge_type(self, start_node: dict, end_node: dict) -> str:
        """
        Classifies the edge type based on node properties.
        """
        z_diff = abs(start_node.get("pos_z", 0) - end_node.get("pos_z", 0))
        z_threshold = self.super_network_config.get("z_level_diff_threshold", 1.0)
        if z_diff > z_threshold:
            return "vertical"

        start_category = start_node.get("category")
        end_category = end_node.get("category")

        if start_category == "CONNECTOR" or end_category == "CONNECTOR":
            return "door"

        return "horizontal"

    def _get_node_color(self, node_name: str) -> str:
        """
        Determines the plotting color for a given node name from config.
        """
        node_props = self.node_defs.get(node_name, {})
        rgb = node_props.get("rgb")
        if rgb and len(rgb) == 3:
            return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
        return "#1f77b4"  # Default Plotly blue

    @abc.abstractmethod
    def plot(
        self,
        graph: nx.Graph,
        output_path: pathlib.Path | None = None,
        title: str = "Network Graph",
        # For Plotly layout, original image width
        graph_width: int | None = None,
        # For Plotly layout, original image height
        graph_height: int | None = None,
        # For SuperNetwork floor labels
        floor_z_map: dict[int, float] | None = None,
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

    def _create_floor_selection_controls(
        self,
        all_z_levels: list[float],
        min_z: float,
        max_z: float,
        floor_z_map_for_labels: dict[int, float] | None = None,
        base_floor_for_labels: int = 0,
    ) -> dict[str, Any]:
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
        z_to_floor_label_map: dict[float, str] = {}
        if floor_z_map_for_labels:
            # Invert floor_z_map_for_labels to map z -> floor_num for easier lookup
            # Handle potential multiple floors at the same Z (unlikely with good input)
            z_to_floor_num: dict[float, list[int]] = {}
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
            slider_steps.append(
                {
                    "label": label,
                    "method": "relayout",
                    "args": [
                        {
                            "scene.zaxis.range": [
                                z_level
                                - self.super_network_config.get("floor_height", 3.0) / 2
                                + 0.1,
                                z_level
                                + self.super_network_config.get("floor_height", 3.0) / 2
                                - 0.1,
                            ]
                        }
                    ],  # View single floor
                }
            )

        # Add a step to show all floors
        slider_steps.append(
            {
                "label": "All Floors",
                "method": "relayout",
                "args": [
                    {
                        "scene.zaxis.range": [
                            min_z
                            - self.super_network_config.get("floor_height", 3.0) * 0.5,
                            max_z
                            + self.super_network_config.get("floor_height", 3.0) * 0.5,
                        ]
                    }
                ],  # View all
            }
        )

        sliders = [
            {
                "active": len(all_z_levels),  # Default to "All Floors"
                "currentvalue": {"prefix": "Current Display: "},
                "pad": {"t": 50},
                "steps": slider_steps,
                "name": "Floor Selection",
            }
        ]
        return {"sliders": sliders}

    def plot(
        self,
        graph: nx.Graph,
        output_path: str | None = None,
        title: str = "3D Network Graph",
        graph_width: int | None = None,
        floor_z_map: dict[int, float] | None = None,
    ):
        if not graph.nodes:
            logger.warning("PlotlyPlotter: Graph has no nodes to plot.")
            return

        nodes_by_name: dict[str, dict[str, list]] = {}
        all_z_coords = [data.get("pos_z", 0) for _, data in graph.nodes(data=True)]

        # Group nodes by their 'name' for creating traces
        for node_id, data in graph.nodes(data=True):
            name = data.get("name", "Unknown")
            if name not in nodes_by_name:
                nodes_by_name[name] = {
                    "x": [],
                    "y": [],
                    "z": [],
                    "hover_text": [],
                    "sizes": [],
                    "ids": [],
                }

            if not self._validate_node_coordinates(data, node_id):
                logger.warning(f"Skipping node {node_id} due to invalid coordinates.")
                continue

            x, y, z = data["pos_x"], data["pos_y"], data["pos_z"]
            plot_x = (
                (graph_width - x)
                if self.plotter_config.get("image_mirror") and graph_width
                else x
            )

            nodes_by_name[name]["x"].append(plot_x)
            nodes_by_name[name]["y"].append(y)
            nodes_by_name[name]["z"].append(z)
            nodes_by_name[name]["ids"].append(node_id)

            hover_label = f"ID: {node_id}<br>Name: {name}<br>CName: {data.get('cname', 'N/A')}<br>Code: {data.get('code', 'N/A')}<br>Pos: ({x:.1f}, {y:.1f}, {z:.1f})"
            if name == "Door":
                door_type = data.get("door_type", "N/A")
                hover_label += f"<br>Door Type: {door_type}"
            nodes_by_name[name]["hover_text"].append(hover_label)

            node_sizes = self.plotter_config.get("node_sizes", {})
            size = node_sizes.get(
                data.get("category", "default"), node_sizes.get("default", 3)
            )
            nodes_by_name[name]["sizes"].append(size)

        # Create node traces
        node_traces = []
        for name, data in nodes_by_name.items():
            node_traces.append(
                go.Scatter3d(
                    x=data["x"],
                    y=data["y"],
                    z=data["z"],
                    mode="markers",
                    marker={
                        "size": data["sizes"],
                        "color": self._get_node_color(name),
                        "opacity": self.plotter_config.get("node_opacity", 0.8),
                    },
                    text=data["hover_text"],
                    hoverinfo="text",
                    name=name,
                    customdata=data["ids"],
                )
            )

        # Prepare edge data
        edge_styles = self._get_edge_style_config()
        edge_data = {key: {"x": [], "y": [], "z": []} for key in edge_styles}

        for u, v in graph.edges():
            node_u, node_v = graph.nodes[u], graph.nodes[v]
            if not self._validate_node_coordinates(
                node_u, u
            ) or not self._validate_node_coordinates(node_v, v):
                continue

            x0, y0, z0 = node_u["pos_x"], node_u["pos_y"], node_u["pos_z"]
            x1, y1, z1 = node_v["pos_x"], node_v["pos_y"], node_v["pos_z"]

            if self.plotter_config.get("image_mirror") and graph_width:
                x0, x1 = graph_width - x0, graph_width - x1

            edge_type = self._classify_edge_type(node_u, node_v)
            edge_data[edge_type]["x"].extend([x0, x1, None])
            edge_data[edge_type]["y"].extend([y0, y1, None])
            edge_data[edge_type]["z"].extend([z0, z1, None])

        # Create edge traces
        edge_traces = []
        for edge_type, data in edge_data.items():
            if data["x"]:
                style = edge_styles[edge_type]
                edge_traces.append(
                    go.Scatter3d(
                        x=data["x"],
                        y=data["y"],
                        z=data["z"],
                        mode="lines",
                        line={"color": style["color"], "width": style["width"]},
                        hoverinfo="none",
                        name=style["name"],
                        showlegend=True,
                    )
                )

        # Create layout
        scene_config = self.plotter_config.get("scene", {})
        aspect_ratio = scene_config.get("aspect_ratio", {"x": 1, "y": 1, "z": 1})
        camera_eye = scene_config.get("camera", {}).get(
            "eye", {"x": 1.25, "y": 1.25, "z": 1.25}
        )

        layout = go.Layout(
            title=title,
            showlegend=True,
            hovermode="closest",
            margin={"b": 20, "l": 5, "r": 5, "t": 40},
            scene={
                "xaxis": {
                    "title": "X",
                },
                "yaxis": {"title": "Y"},
                "zaxis": {"title": "Z (Floor)"},
                "aspectmode": "manual",
                "aspectratio": {
                    "x": aspect_ratio["x"],
                    "y": aspect_ratio["y"],
                    "z": aspect_ratio["z"],
                },
                "camera": {
                    "eye": {
                        "x": camera_eye["x"],
                        "y": camera_eye["y"],
                        "z": camera_eye["z"],
                    }
                },
            },
            legend={
                "orientation": "v",
                "x": 0.02,
                "y": 1.0,
                "xanchor": "left",
                "yanchor": "top",
                "bgcolor": "rgba(255, 255, 255, 0.7)",
            },
        )

        unique_z = sorted({z for z in all_z_coords if z is not None})
        if len(unique_z) > 1:
            min_z, max_z = min(unique_z), max(unique_z)
            layout.update(
                self._create_floor_selection_controls(
                    all_z_levels=unique_z,
                    min_z=min_z,
                    max_z=max_z,
                    floor_z_map_for_labels=floor_z_map,
                )
            )

        fig = go.Figure(data=node_traces + edge_traces, layout=layout)

        if output_path:
            p = pathlib.Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(p), config=self.plotter_config.get("plotly_config"))
        else:
            fig.show()
