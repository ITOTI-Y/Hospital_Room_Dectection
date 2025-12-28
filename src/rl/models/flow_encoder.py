"""
Flow-Aware Encoder for dynamic adaptation to patient flow changes.

Key innovation: Separates "department attributes" (static) from "flow demands" (dynamic),
enabling the model to adapt to changing patient patterns without full retraining.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.data import Batch, Data


class FlowAwareEncoder(nn.Module):
    """
    Flow-aware encoder that explicitly encodes patient flow demands into node representations.

    This enables the model to:
    1. Understand which department pairs have high traffic
    2. Adapt when flow patterns change (e.g., seasonal variations)
    3. Prioritize swaps that affect high-flow connections
    """

    def __init__(
        self,
        dept_attr_dim: int,
        flow_embed_dim: int,
        hidden_dim: int,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Args:
            dept_attr_dim: Dimension of department attributes (service_time, area, etc.)
            flow_embed_dim: Dimension of flow input (typically max_departments)
            hidden_dim: Hidden dimension for encodings
            n_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        # Department attribute encoder (static features)
        self.dept_encoder = nn.Sequential(
            nn.Linear(dept_attr_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # Flow demand encoder (dynamic features)
        # Input: each department's flow weights with all other departments
        self.flow_encoder = nn.Sequential(
            nn.Linear(flow_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # Cross-attention: let flow information influence department representations
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
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
        dept_attrs: torch.Tensor,
        flow_matrix: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            dept_attrs: (batch, n_depts, attr_dim) - department attributes
            flow_matrix: (batch, n_depts, n_depts) - patient flow demands
            node_mask: (batch, n_depts) - 1 for valid nodes, 0 for padding

        Returns:
            (batch, n_depts, hidden_dim) - flow-aware department embeddings
        """
        # Encode static attributes
        dept_embeds = self.dept_encoder(dept_attrs)  # (batch, n_depts, hidden)

        # Encode flow demands - each department's flow profile
        flow_embeds = self.flow_encoder(flow_matrix)  # (batch, n_depts, hidden)

        # Cross-attention: departments attend to flow information
        # Query: department, Key/Value: flow
        attn_mask = ~node_mask.bool()  # True means ignore
        attended, _ = self.cross_attention(
            query=dept_embeds,
            key=flow_embeds,
            value=flow_embeds,
            key_padding_mask=attn_mask,
        )

        # Fusion
        fused = self.fusion(torch.cat([dept_embeds, attended], dim=-1))

        return fused  # (batch, n_depts, hidden)


class AdaptiveLayoutEncoder(nn.Module):
    """
    Complete layout encoder with flow-awareness and spatial GNN.

    Architecture:
    1. FlowAwareEncoder: encodes department attrs + flow demands
    2. Spatial GNN: propagates information along travel connections
    """

    def __init__(
        self,
        n_dept_attrs: int = 7,
        max_depts: int = 100,
        hidden_dim: int = 128,
        gnn_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Args:
            n_dept_attrs: Number of department attributes
            max_depts: Maximum number of departments
            hidden_dim: Hidden dimension
            gnn_layers: Number of GNN layers
            n_heads: Attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.flow_encoder = FlowAwareEncoder(
            dept_attr_dim=n_dept_attrs,
            flow_embed_dim=max_depts,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            dropout=dropout,
        )

        # Spatial relationship GNN
        self.gnn_layers = nn.ModuleList([
            GATConv(
                hidden_dim,
                hidden_dim // n_heads,
                heads=n_heads,
                concat=True,
                dropout=dropout,
            )
            for _ in range(gnn_layers)
        ])

        self.output_dim = hidden_dim

    def forward(
        self,
        dept_attrs: torch.Tensor,
        flow_matrix: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            dept_attrs: (batch, n_depts, attr_dim)
            flow_matrix: (batch, n_depts, n_depts)
            edge_index: (batch, 2, n_edges) or (2, n_edges) for single
            edge_weight: (batch, n_edges) or (n_edges,)
            node_mask: (batch, n_depts)

        Returns:
            (batch, n_depts, hidden_dim) - encoded node features
        """
        # Flow-aware encoding
        x = self.flow_encoder(dept_attrs, flow_matrix, node_mask)

        batch_size, n_nodes, hidden = x.shape

        # Process each batch item through GNN
        # Note: In production, use PyG's Batch mechanism for efficiency
        outputs = []
        for b in range(batch_size):
            h = x[b]  # (n_nodes, hidden)

            # Handle edge_index dimensions
            if edge_index.dim() == 3:
                ei = edge_index[b]  # (2, n_edges)
                ew = edge_weight[b] if edge_weight.dim() == 2 else edge_weight
            else:
                ei = edge_index
                ew = edge_weight

            # Filter valid edges (non-negative indices)
            valid_edge_mask = (ei[0] >= 0) & (ei[1] >= 0)
            ei_valid = ei[:, valid_edge_mask]

            if ei_valid.shape[1] > 0:
                for gnn in self.gnn_layers:
                    h = F.relu(gnn(h, ei_valid))

            outputs.append(h)

        return torch.stack(outputs, dim=0)  # (batch, n_nodes, hidden)


class FlowConditionedPolicy(nn.Module):
    """
    Policy network conditioned on flow patterns.

    Uses flow matrix as additional context for action selection,
    enabling better adaptation to different patient flow scenarios.
    """

    def __init__(
        self,
        encoder: AdaptiveLayoutEncoder,
        action_hidden_dim: int = 256,
        max_depts: int = 100,
    ):
        super().__init__()

        self.encoder = encoder
        hidden_dim = encoder.output_dim

        # Flow context aggregator
        self.flow_context = nn.Sequential(
            nn.Linear(max_depts * max_depts, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Action head (selects swap pair)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, action_hidden_dim),
            nn.ReLU(),
            nn.Linear(action_hidden_dim, max_depts),
        )

    def forward(
        self,
        dept_attrs: torch.Tensor,
        flow_matrix: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute action logits.

        Returns:
            (batch, max_depts) - logits for first swap position
        """
        # Encode layout with flow awareness
        node_embeds = self.encoder(
            dept_attrs, flow_matrix, edge_index, edge_weight, node_mask
        )

        # Global flow context
        batch_size = flow_matrix.shape[0]
        flow_flat = flow_matrix.view(batch_size, -1)
        flow_ctx = self.flow_context(flow_flat)  # (batch, hidden)

        # Combine node embeddings with flow context for action selection
        flow_ctx_expanded = flow_ctx.unsqueeze(1).expand_as(node_embeds)
        combined = torch.cat([node_embeds, flow_ctx_expanded], dim=-1)

        # Action logits
        logits = self.action_head(combined)  # (batch, n_nodes, max_depts)

        return logits.mean(dim=1)  # Simplified: average over nodes


class FlowAwareGCNEncoder(nn.Module):
    """
    Flow-aware GCN encoder that's compatible with the existing GCNEncoder interface.

    This encoder combines:
    1. Flow-aware attention to incorporate patient flow patterns
    2. GCN layers for spatial relationship modeling

    Key difference from GCNEncoder: accepts flow_matrix as additional input
    to enable dynamic adaptation to changing patient flows.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        max_departments: int = 100,
        num_layers: int = 3,
        dropout: float = 0.1,
        n_attention_heads: int = 4,
        use_flow_attention: bool = True,
    ):
        """
        Args:
            input_dim: Input feature dimension (from FeatureProcessor)
            hidden_dims: List of hidden dimensions for GCN layers
            output_dim: Output embedding dimension
            max_departments: Maximum number of departments (for flow matrix)
            num_layers: Number of GCN layers
            dropout: Dropout rate
            n_attention_heads: Number of attention heads for flow attention
            use_flow_attention: Whether to use flow-aware attention
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.max_departments = max_departments
        self.use_flow_attention = use_flow_attention

        if len(hidden_dims) != num_layers - 1:
            raise ValueError(
                f"Length of hidden_dims must be {num_layers - 1}, got {len(hidden_dims)}"
            )

        self.dims = [input_dim, *hidden_dims, output_dim]

        # Flow encoder (projects flow matrix row to hidden dim)
        if use_flow_attention:
            self.flow_encoder = nn.Sequential(
                nn.Linear(max_departments, output_dim),
                nn.ReLU(),
                nn.LayerNorm(output_dim),
                nn.Dropout(dropout),
            )

            # Cross-attention for flow conditioning
            self.flow_attention = nn.MultiheadAttention(
                embed_dim=output_dim,
                num_heads=n_attention_heads,
                dropout=dropout,
                batch_first=True,
            )

            # Projection to match dimensions if needed
            if input_dim != output_dim:
                self.input_projection = nn.Linear(input_dim, output_dim)
            else:
                self.input_projection = nn.Identity()

        # GCN layers
        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for i in range(num_layers):
            in_dim = self.dims[i]
            out_dim = self.dims[i + 1]

            self.convs.append(
                GCNConv(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    add_self_loops=True,
                    normalize=True,
                )
            )
            self.layer_norms.append(nn.LayerNorm(out_dim))

        self.dropout = nn.Dropout(dropout)

        # Residual projections
        self.residual_projections = nn.ModuleList()
        for i in range(num_layers):
            in_dim = self.dims[i]
            out_dim = self.dims[i + 1]
            if in_dim != out_dim:
                self.residual_projections.append(nn.Linear(in_dim, out_dim, bias=False))
            else:
                self.residual_projections.append(nn.Identity())

        # Fusion layer for combining flow-attended and GCN features
        if use_flow_attention:
            self.fusion = nn.Sequential(
                nn.Linear(output_dim * 2, output_dim),
                nn.ReLU(),
                nn.LayerNorm(output_dim),
            )

        self._reset_parameters()

    def _reset_parameters(self):
        for conv in self.convs:
            if hasattr(conv, "lin") and isinstance(conv.lin, nn.Linear):
                nn.init.xavier_uniform_(conv.lin.weight)
                if conv.lin.bias is not None:
                    nn.init.zeros_(conv.lin.bias)

        for proj in self.residual_projections:
            if isinstance(proj, nn.Linear):
                nn.init.xavier_uniform_(proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Standard GCN forward (without flow matrix)."""
        for i, conv in enumerate(self.convs):
            identity = x
            x = conv(x, edge_index, edge_weight)
            x = self.layer_norms[i](x)

            if i < self.num_layers - 1:
                x = torch.relu(x)
                x = self.dropout(x)

            identity = self.residual_projections[i](identity)
            x = x + identity

        return x

    def forward_with_flow(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor,
        flow_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with flow matrix conditioning.

        Args:
            x: Node features (batch, num_nodes, input_dim)
            edge_index: Edge indices (batch, 2, E_max)
            edge_weight: Edge weights (batch, E_max)
            node_mask: Node validity mask (batch, num_nodes)
            edge_mask: Edge validity mask (batch, E_max)
            flow_matrix: Patient flow matrix (batch, num_nodes, num_nodes)

        Returns:
            Node embeddings (batch, num_nodes, output_dim)
        """
        batch_size, num_nodes, _ = x.size()

        # Step 1: Flow-aware attention (if enabled)
        if self.use_flow_attention:
            # Project input features
            x_proj = self.input_projection(x)  # (batch, num_nodes, output_dim)

            # Encode flow patterns
            flow_embeds = self.flow_encoder(flow_matrix)  # (batch, num_nodes, output_dim)

            # Cross-attention: nodes attend to flow information
            attn_mask = ~node_mask.bool()  # True means ignore
            flow_attended, _ = self.flow_attention(
                query=x_proj,
                key=flow_embeds,
                value=flow_embeds,
                key_padding_mask=attn_mask,
            )
        else:
            flow_attended = None

        # Step 2: GCN encoding (using PyG Batch)
        data_list = []
        for i in range(batch_size):
            valid_nodes = node_mask[i].bool()
            valid_edges = edge_mask[i].bool()

            node_feat = x[i][valid_nodes]
            valid_edge_index = edge_index[i][:, valid_edges]
            valid_edge_weight = edge_weight[i][valid_edges]

            data = Data(
                x=node_feat,
                edge_index=valid_edge_index,
                edge_attr=valid_edge_weight,
            )
            data_list.append(data)

        batch_data = Batch.from_data_list(data_list)

        # Run through GCN layers
        gcn_output = self.forward(
            x=batch_data.x,
            edge_index=batch_data.edge_index,
            edge_weight=batch_data.edge_attr,
        )

        # Unpack batch back to padded tensor
        output = torch.zeros(
            batch_size, num_nodes, self.output_dim, device=x.device, dtype=x.dtype
        )

        ptr = 0
        for i in range(batch_size):
            num_valid = int(node_mask[i].sum().item())
            valid_nodes = node_mask[i].bool()
            output[i][valid_nodes] = gcn_output[ptr : ptr + num_valid]
            ptr += num_valid

        # Step 3: Fuse flow-attended and GCN features
        if self.use_flow_attention and flow_attended is not None:
            fused = self.fusion(torch.cat([output, flow_attended], dim=-1))
            return fused

        return output

    def forward_batch(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor,
        flow_matrix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Batch forward pass (compatible interface with GCNEncoder.forward_batch).

        If flow_matrix is provided, uses flow-aware encoding.
        Otherwise, falls back to standard GCN encoding.
        """
        if flow_matrix is not None and self.use_flow_attention:
            return self.forward_with_flow(
                x, edge_index, edge_weight, node_mask, edge_mask, flow_matrix
            )

        # Standard GCN encoding (backward compatible)
        batch_size, num_nodes, _ = x.size()

        data_list = []
        for i in range(batch_size):
            valid_nodes = node_mask[i].bool()
            valid_edges = edge_mask[i].bool()

            node_feat = x[i][valid_nodes]
            valid_edge_index = edge_index[i][:, valid_edges]
            valid_edge_weight = edge_weight[i][valid_edges]

            data = Data(
                x=node_feat,
                edge_index=valid_edge_index,
                edge_attr=valid_edge_weight,
            )
            data_list.append(data)

        batch_data = Batch.from_data_list(data_list)

        node_embeddings = self.forward(
            x=batch_data.x,
            edge_index=batch_data.edge_index,
            edge_weight=batch_data.edge_attr,
        )

        output = torch.zeros(
            batch_size, num_nodes, self.output_dim, device=x.device, dtype=x.dtype
        )

        ptr = 0
        for i in range(batch_size):
            num_valid = int(node_mask[i].sum().item())
            valid_nodes = node_mask[i].bool()
            output[i][valid_nodes] = node_embeddings[ptr : ptr + num_valid]
            ptr += num_valid

        return output
