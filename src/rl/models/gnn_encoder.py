import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
from tianshou.data import Batch as TianshouBatch
from typing import Optional, List


class GCNEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        if len(hidden_dims) != num_layers - 1:
            raise ValueError(
                f"Length of hidden_dim must be {num_layers - 1}, got {len(hidden_dims)}"
            )

        self.dims = [input_dim] + hidden_dims + [output_dim]

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

        self.residual_projections = nn.ModuleList()
        for i in range(num_layers):
            in_dim = self.dims[i]
            out_dim = self.dims[i + 1]
            if in_dim != out_dim:
                self.residual_projections.append(nn.Linear(in_dim, out_dim, bias=False))
            else:
                self.residual_projections.append(nn.Identity())

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
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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

    def forward_batch(
        self,
        x: TianshouBatch | torch.Tensor,
        edge_index: TianshouBatch | torch.Tensor,
        edge_weight: TianshouBatch | torch.Tensor,
        node_mask: TianshouBatch | torch.Tensor,
        edge_mask: TianshouBatch | torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_nodes, feat_dim = x.size()

        data_list = []
        for i in range(batch_size):
            valid_nodes = node_mask[i].bool()
            valid_edges = edge_mask[i].bool()

            node_feat = x[i][valid_nodes]

            valid_edge_index = edge_index[i][:, valid_edges]
            valid_edge_weight = edge_weight[i][valid_edges]

            data = Data(
                x=node_feat, edge_index=valid_edge_index, edge_attr=valid_edge_weight
            )
            data_list.append(data)

        batch_data = Batch.from_data_list(data_list)

        node_embeddings = self.forward(
            x=batch_data.x,  # type: ignore
            edge_index=batch_data.edge_index,  # type: ignore
            edge_weight=batch_data.edge_attr,  # type: ignore
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
