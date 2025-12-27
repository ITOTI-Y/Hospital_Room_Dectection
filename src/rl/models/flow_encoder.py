"""Flow-aware encoder for dynamic adaptation to patient flow changes.

This module implements encoders that explicitly separate:
- Static department attributes (service time, area, etc.)
- Dynamic flow demands (patient flow matrix)

This separation enables the model to quickly adapt when flow patterns change.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class FlowAwareEncoder(nn.Module):
    """Flow-aware encoder that separates static and dynamic features.

    Key innovation: Separates "inherent department attributes" from "flow demands",
    allowing the model to adapt to flow changes without retraining.

    Args:
        dept_attr_dim: Dimension of department attributes (service_time, area, etc.)
        flow_embed_dim: Dimension of flow embedding (typically n_depts)
        hidden_dim: Hidden dimension for embeddings
        n_heads: Number of attention heads for cross-attention
    """

    def __init__(
        self,
        dept_attr_dim: int,
        flow_embed_dim: int,
        hidden_dim: int,
        n_heads: int = 4,
    ):
        super().__init__()

        # Department attribute encoder (static features)
        self.dept_encoder = nn.Sequential(
            nn.Linear(dept_attr_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        # Flow demand encoder (dynamic features)
        # Input: flow weight vector between each dept and all other depts
        self.flow_encoder = nn.Sequential(
            nn.Linear(flow_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        # Cross-attention: let flow information modulate department representation
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            batch_first=True,
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        self.output_dim = hidden_dim

    def forward(
        self,
        dept_attrs: torch.Tensor,  # (batch, n_depts, attr_dim)
        flow_matrix: torch.Tensor,  # (batch, n_depts, n_depts)
        node_mask: torch.Tensor,  # (batch, n_depts)
    ) -> torch.Tensor:
        """Encode departments with flow-aware representations.

        Args:
            dept_attrs: Static department attributes
            flow_matrix: Patient flow demand matrix
            node_mask: Valid node mask (1 for valid, 0 for padding)

        Returns:
            Flow-aware node embeddings (batch, n_depts, hidden_dim)
        """
        # Encode static attributes
        dept_embeds = self.dept_encoder(dept_attrs)  # (batch, n_depts, hidden)

        # Encode flow demands - each department's flow profile
        flow_embeds = self.flow_encoder(flow_matrix)  # (batch, n_depts, hidden)

        # Cross-attention: departments attend to flow patterns
        # Query: department, Key/Value: flow
        attn_mask = ~node_mask.bool()  # True = ignore
        attended, _ = self.cross_attention(
            query=dept_embeds,
            key=flow_embeds,
            value=flow_embeds,
            key_padding_mask=attn_mask,
        )

        # Fuse department and flow representations
        fused = self.fusion(torch.cat([dept_embeds, attended], dim=-1))

        return fused  # (batch, n_depts, hidden)


class AdaptiveLayoutEncoder(nn.Module):
    """Complete layout encoder with dynamic adaptation support.

    Combines flow-aware encoding with spatial relation modeling via GNN.

    Args:
        n_dept_attrs: Number of department attribute features
        max_depts: Maximum number of departments (for flow embedding)
        hidden_dim: Hidden dimension
        gnn_layers: Number of GNN layers
        gnn_heads: Number of attention heads per GAT layer
    """

    def __init__(
        self,
        n_dept_attrs: int = 4,  # service_time, area, x, y
        max_depts: int = 100,
        hidden_dim: int = 128,
        gnn_layers: int = 3,
        gnn_heads: int = 4,
    ):
        super().__init__()

        self.flow_encoder = FlowAwareEncoder(
            dept_attr_dim=n_dept_attrs,
            flow_embed_dim=max_depts,
            hidden_dim=hidden_dim,
        )

        # Spatial relation GNN using Graph Attention
        self.gnn_layers = nn.ModuleList()
        for _ in range(gnn_layers):
            self.gnn_layers.append(
                GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // gnn_heads,
                    heads=gnn_heads,
                    concat=True,
                    dropout=0.1,
                )
            )

        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(gnn_layers)]
        )

        self.output_dim = hidden_dim

    def forward(
        self,
        dept_attrs: torch.Tensor,  # (batch, n_nodes, n_dept_attrs)
        flow_matrix: torch.Tensor,  # (batch, n_nodes, n_nodes)
        edge_index: torch.Tensor,  # (batch, 2, n_edges)
        edge_weight: torch.Tensor,  # (batch, n_edges)
        node_mask: torch.Tensor,  # (batch, n_nodes)
    ) -> torch.Tensor:
        """Encode layout with flow and spatial information.

        Args:
            dept_attrs: Department attributes
            flow_matrix: Patient flow matrix
            edge_index: Graph edges (spatial connectivity)
            edge_weight: Edge weights
            node_mask: Valid node mask

        Returns:
            Node embeddings (batch, n_nodes, hidden_dim)
        """
        # Flow-aware encoding
        x = self.flow_encoder(dept_attrs, flow_matrix, node_mask)

        # GNN propagation for spatial information
        batch_size, n_nodes, hidden = x.shape

        # Process each graph in batch separately
        # Note: In production, should use PyG's Batch for efficiency
        outputs = []
        for b in range(batch_size):
            h = x[b]  # (n_nodes, hidden)
            ei = edge_index[b]  # (2, n_edges)
            ew = edge_weight[b]  # (n_edges,)

            # Apply GNN layers
            for layer, norm in zip(self.gnn_layers, self.layer_norms):
                h_new = layer(h, ei, ew)
                h_new = norm(h_new)
                h = F.relu(h_new) + h  # Residual connection

            outputs.append(h)

        return torch.stack(outputs, dim=0)  # (batch, n_nodes, hidden)


class FlowMatrixExtractor(nn.Module):
    """Helper module to extract flow matrix from observation.

    Converts edge-based flow representation to dense flow matrix.
    """

    def __init__(self, max_depts: int):
        super().__init__()
        self.max_depts = max_depts

    def forward(
        self,
        edge_index: torch.Tensor,  # (batch, 2, n_edges)
        edge_weight: torch.Tensor,  # (batch, n_edges)
        edge_mask: torch.Tensor,  # (batch, n_edges)
    ) -> torch.Tensor:
        """Convert edge representation to flow matrix.

        Args:
            edge_index: Edge indices
            edge_weight: Edge weights (flow demands)
            edge_mask: Valid edge mask

        Returns:
            Flow matrix (batch, max_depts, max_depts)
        """
        batch_size = edge_index.size(0)
        device = edge_index.device

        flow_matrix = torch.zeros(
            batch_size, self.max_depts, self.max_depts, device=device
        )

        for b in range(batch_size):
            ei = edge_index[b]  # (2, n_edges)
            ew = edge_weight[b]  # (n_edges,)
            em = edge_mask[b].bool()  # (n_edges,)

            # Only consider valid edges
            valid_ei = ei[:, em]
            valid_ew = ew[em]

            # Fill flow matrix (symmetric)
            src, dst = valid_ei[0], valid_ei[1]
            flow_matrix[b, src, dst] = valid_ew
            flow_matrix[b, dst, src] = valid_ew  # Ensure symmetry

        return flow_matrix
