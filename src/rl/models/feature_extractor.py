import torch
import torch.nn as nn
from typing import Optional


class EmbeddingLayer(nn.Module):
    def __init__(self, num_categories: int, embedding_dim: int, padding_idx: int = -1):
        """
        num_categories is equal max_departments
        value == padding_idx position will be set to zero and ignored in the loss computation
        """
        super().__init__()
        self.num_categories = num_categories
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(
            num_embeddings=self.num_categories,
            embedding_dim=self.embedding_dim,
            padding_idx=padding_idx,
        )

        nn.init.xavier_uniform_(self.embedding.weight)
        if padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[padding_idx].fill_(0)

    def forward(self, categories_ids: torch.Tensor) -> torch.Tensor:
        """

        Args:
            categories_ids (torch.Tensor): categories indices, shape (batch_size, num_nodes[max_departments])

        Returns:
            embedded (torch.Tensor): embedded features, shape (batch_size, num_nodes, embedding_dim)
        """

        categories_ids = categories_ids.long()
        embedded = self.embedding(
            categories_ids
        )  # (batch_size, num_nodes, embedding_dim)
        return embedded


class FeatureProcess(nn.Module):
    """
    FeatureProcess : intergrate categorical and numerical feature
    """

    def __init__(
        self,
        num_categories: int,
        embedding_dim: int,
        numerical_feat_dim: int,
        numerical_hidden_dim: Optional[int] = None,
        padding_idx: int = -1,
    ):
        super().__init__()
        self.num_categories = num_categories
        self.embedding_dim = embedding_dim
        self.numerical_feat_dim = numerical_feat_dim

        self.embedding_layer = EmbeddingLayer(
            num_categories=self.num_categories,
            embedding_dim=self.embedding_dim,
            padding_idx=padding_idx,
        )

        self.numerical_hidden_dim = numerical_hidden_dim or embedding_dim
        self.numerical_projection = nn.Sequential(
            nn.Linear(self.numerical_feat_dim, self.numerical_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.numerical_hidden_dim),
        )
        self.output_dim = embedding_dim + self.numerical_hidden_dim

    def forward(
        self,
        categorical_feat: torch.Tensor,
        numerical_feat: torch.Tensor,
    ) -> torch.Tensor:
        
        embedded_categorical = self.embedding_layer(
            categorical_feat
        )  # (batch_size, num_nodes, embedding_dim)

        projected_numerical = self.numerical_projection(
            numerical_feat
        )  # (batch_size, num_nodes, numerical_hidden_dim)

        conbined_features = torch.cat(
            [embedded_categorical, projected_numerical], dim=-1
        )  # (batch_size, num_nodes, embedding_dim + numerical_hidden_dim)

        return conbined_features