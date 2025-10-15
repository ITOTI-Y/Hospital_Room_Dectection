import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn.aggr import AttentionalAggregation
from typing import Optional, Literal
from loguru import logger

class GlobalPooling(nn.Module):
    def __init__(
        self,
        pooling_type: Literal["mean", "max", "sum", "attention"] = "mean",
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.pooling_type = pooling_type

        if pooling_type == "mean":
            self.pool = global_mean_pool
        elif pooling_type == "max":
            self.pool = global_max_pool
        elif pooling_type == "sum":
            self.pool = global_add_pool
        elif pooling_type == "attention":
            if hidden_dim is None:
                logger.error("hidden_dim must be provided for attention pooling")
                raise ValueError("hidden_dim must be provided for attention pooling")
            gate_nn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1),
            )
            self.pool = AttentionalAggregation(gate_nn)
        else:
            logger.error(f"Unsupported pooling type: {pooling_type}")
            raise ValueError(f"Unsupported pooling type: {pooling_type}")

    def forward(
            self,
            node_embeddings: torch.Tensor,
            node_mask: torch.Tensor,
            batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        if batch is not None:
            pooled = self.pool(node_embeddings, batch) # node_embeddings: (num_nodes, hidden_dim), batch: (num_nodes,)
        else:
            batch_size, num_nodes, hidden_dim = node_embeddings.size() # (batch_size, num_nodes, hidden_dim)

            batch_indices = torch.arange(batch_size, device=node_embeddings.device)
            batch_indices = batch_indices.unsqueeze(-1).expand(-1, num_nodes) # (batch_size, num_nodes)

            valid_mask = node_mask.bool()
            valid_node_embeddings = node_embeddings[valid_mask] # (num_valid_nodes, hidden_dim)
            valid_batch = batch_indices[valid_mask] # (num_valid_nodes,)

            pooled = self.pool(valid_node_embeddings, valid_batch) # (batch_size, hidden_dim)
        
        return pooled
    

class ValueNet(nn.Module):

    def __init__(
            self,
            node_embedding_dim: int,
            value_hidden_dim: int = 256,
            num_layers: int = 3,
            pooling_type: Literal["mean", "max", "sum", "attention"] = "mean",
            dropout: float = 0.1,
    ):
        super().__init__()

        self.node_embedding_dim = node_embedding_dim
        self.value_hidden_dim = value_hidden_dim
        self.num_layers = num_layers

        self.global_pooling = GlobalPooling(
            pooling_type=pooling_type,
            hidden_dim=node_embedding_dim if pooling_type == "attention" else None,
        )

        layers = []
        input_dim = node_embedding_dim

        for _ in range(num_layers):
            layers.extend([
                nn.Linear(input_dim, value_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(value_hidden_dim),
            ])
            input_dim = value_hidden_dim

        layers.append(nn.Linear(value_hidden_dim, 1))

        self.value_head = nn.Sequential(*layers)

        self._reset_parameters()

    def _reset_parameters(self):
        for module in self.value_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain('relu'))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
            self,
            node_embeddings: torch.Tensor,
            node_mask: torch.Tensor,
            batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        global_state = self.global_pooling(
            node_embeddings,
            node_mask,
            batch=batch,
        )

        state_value = self.value_head(global_state) # (batch_size, 1)

        return state_value
    
    def get_value(
            self,
            node_embeddings: torch.Tensor,
            node_mask: torch.Tensor,
            batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        value = self.forward(
            node_embeddings,
            node_mask,
            batch=batch,
        ) # (batch_size, 1)

        return value.squeeze(-1) # (batch_size,)