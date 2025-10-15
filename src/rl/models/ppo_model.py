import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Tuple, Literal, Dict, Union, Any
from tianshou.data import Batch

from .feature_extractor import FeatureProcessor
from .gnn_encoder import GCNEncoder
from .policy_net import AutoregressiveActor
from .value_net import ValueNet

class LayoutOptimizationModel(nn.Module):
    def __init__(
        self,
        num_categories: int,
        embedding_dim: int,
        numerical_feat_dim: int,
        numerical_hidden_dim: Optional[int] = None,

        gnn_hidden_dims: List[int] = [128, 128],
        gnn_output_dim: int = 256,
        gnn_num_layers: int = 3,
        gnn_dropout: float = 0.1,

        actor_hidden_dim: int = 128,
        actor_dropout: float = 0.1,

        value_hidden_dim: int = 256,
        value_num_layers: int = 3,
        value_pooling_type: Literal["mean", "max", "sum", "attention"] = "mean",
        value_dropout: float = 0.1,

        device: Optional[torch.device] = torch.device("cpu"),
    ):
        super().__init__()
        self.device = device

        self.feature_processor = FeatureProcessor(
            num_categories=num_categories,
            embedding_dim=embedding_dim,
            numerical_feat_dim=numerical_feat_dim,
            numerical_hidden_dim=numerical_hidden_dim,
            padding_idx=-1,
        )

        self.gnn_encoder = GCNEncoder(
            input_dim=self.feature_processor.output_dim,
            hidden_dims=gnn_hidden_dims,
            output_dim=gnn_output_dim,
            num_layers=gnn_num_layers,
            dropout=gnn_dropout,
        )

        self.actor = AutoregressiveActor(
            node_hidden_dim=gnn_output_dim,
            actor_hidden_dim=actor_hidden_dim,
            dropout=actor_dropout,
        )

        self.critic = ValueNet(
            node_embedding_dim=gnn_output_dim,
            value_hidden_dim=value_hidden_dim,
            num_layers=value_num_layers,
            pooling_type=value_pooling_type,
            dropout=value_dropout,
        )

        self.to(self.device)

    def _prepare_inputs(
            self,
            obs: Union[Dict, Batch],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(obs, Batch):
            x_categorical = obs.x_categorical
            x_numerical = obs.x_numerical
            edge_index = obs.edge_index
            edge_weight = obs.edge_weight
            node_mask = obs.node_mask
            edge_mask = obs.edge_mask
        else:
            x_categorical = obs["x_categorical"]
            x_numerical = obs["x_numerical"]
            edge_index = obs["edge_index"]
            edge_weight = obs["edge_weight"]
            node_mask = obs["node_mask"]
            edge_mask = obs["edge_mask"]

        x_categorical = torch.as_tensor(x_categorical, device=self.device, dtype=torch.long)
        x_numerical = torch.as_tensor(x_numerical, device=self.device, dtype=torch.float32)
        edge_index = torch.as_tensor(edge_index, device=self.device, dtype=torch.long)
        edge_weight = torch.as_tensor(edge_weight, device=self.device, dtype=torch.float32)
        node_mask = torch.as_tensor(node_mask, device=self.device, dtype=torch.float32)
        edge_mask = torch.as_tensor(edge_mask, device=self.device, dtype=torch.float32)

        return x_categorical, x_numerical, edge_index, edge_weight, node_mask, edge_mask
    
    def _encode_observations(
            self,
            obs: Union[Dict, Batch],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_categorical, x_numerical, edge_index, edge_weight, node_mask, edge_mask = self._prepare_inputs(obs)

        node_features = self.feature_processor(x_categorical, x_numerical)

        node_embeddings = self.gnn_encoder.forward_batch(
            x=node_features,
            edge_index=edge_index,
            edge_weight=edge_weight,
            node_mask=node_mask,
            edge_mask=edge_mask
        )

        return node_embeddings, node_mask

    def forward(
            self,
            obs: Union[Dict, Batch],
            state: Optional[Any] = None,
            **kwargs
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Any]:

        node_embeddings, node_mask = self._encode_observations(obs)

        action1, action2 = self.actor(
            node_embeddings=node_embeddings,
            node_mask=node_mask,
            deterministic=kwargs.get("deterministic", False),
        )

        return (action1, action2), state
    
    def get_action_log_prob(
            self,
            obs: Union[Dict, Batch],
            action: np.ndarray,
            **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            obs (Union[Dict, Batch]): Observation from the environment.
            action (np.ndarray): Action taken shape (batch_size, 2).

        Returns:
            log_prob (torch.Tensor): Joint log probability of action1 and action2, shape (batch_size,).
            entryopy (torch.Tensor): Sum of entropies of action1 and action2 distributions, shape (batch_size,).
        """

        node_embeddings, node_mask = self._encode_observations(obs)

        actions = torch.as_tensor(action, device=self.device, dtype=torch.long)
        action1 = actions[:, 0]
        action2 = actions[:, 1]

        _, _, log_prob, entryopy = self.actor.get_log_prob_and_entropy(
            node_embeddings=node_embeddings,
            node_mask=node_mask,
            action1=action1,
            action2=action2,
        )

        return log_prob, entryopy
    
    def get_value(
            self,
            obs: Union[Dict, Batch],
            **kwargs: Any,
    ) -> torch.Tensor:
        
        node_embeddings, node_mask = self._encode_observations(obs)

        value = self.critic.get_value(
            node_embeddings=node_embeddings,
            node_mask=node_mask,
        ) # (batch_size)

        return value