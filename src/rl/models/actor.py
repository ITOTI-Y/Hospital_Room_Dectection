import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Optional, Tuple


class AutoregressiveActor(nn.Module):
    def __init__(
        self,
        node_hidden_dim: int,
        actor_hidden_dim: int,
        dropout: float = 0.1,
    ):
        """
        Autoregression Policy Network, used to select two departments to swap.

        Args:
            node_hidden_dim (int): GNN output dimension
            action_hidden_dim (int): Action MLP hidden dimension
            dropout (float, optional): Dropout rate
        """

        super().__init__()

        self.node_hidden_dim: int = node_hidden_dim
        self.actor_hidden_dim: int = actor_hidden_dim

        self.first_action_head = nn.Sequential(
            nn.Linear(node_hidden_dim, actor_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(actor_hidden_dim),
            nn.Linear(actor_hidden_dim, actor_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(actor_hidden_dim // 2, 1),
        )

        self.second_action_head = nn.Sequential(
            nn.Linear(node_hidden_dim * 2, actor_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(actor_hidden_dim),
            nn.Linear(actor_hidden_dim, actor_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(actor_hidden_dim // 2, 1),
        )

    def _create_masked_distribution(
            self,
            logits: torch.Tensor,
            mask: torch.Tensor
    ) -> Categorical:
        
        mask_value = torch.finfo(logits.dtype).min
        masked_logits = logits.masked_fill(mask == 0, mask_value)
        
        dist = Categorical(logits=masked_logits)
        return dist

    def forward(
        self,
        node_embeddings: torch.Tensor,
        node_mask: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[Categorical], Optional[Categorical]]:
        """
        Forward, execute autoregressive two action select

        Args:
            node_embeddings (torch.Tensor): GNN node embeddings, shape (batch_size, num_nodes, node_hidden_dim)
            node_mask (torch.Tensor): Node mask, shape (batch_size, num_nodes) (1 for valid nodes, 0 for padding nodes)

        Returns:
            action1: torch.Tensor: First selected action (node index), shape (batch_size,)
            action2: torch.Tensor: Second selected action (node index), shape (batch_size,)
            log_prob1: torch.Tensor: Log probability of action1, shape (batch_size,)
            log_prob2: torch.Tensor: Log probability of action2, shape (batch_size,)
            dist1: Optional[Categorical]: Action distribution for action1, None if deterministic is True
            dist2: Optional[Categorical]: Action distribution for action2, None if deterministic is
        """

        batch_size, num_nodes, _ = node_embeddings.size()

        logits1: torch.Tensor = self.first_action_head(node_embeddings).squeeze(
            -1
        )  # (batch_size, num_nodes)

        dist1 = self._create_masked_distribution(logits1, node_mask)
        
        if deterministic:
            action1 = dist1.probs.argmax(dim=-1)  #type: ignore (batch_size,)
        else:
            action1 = dist1.sample()  # (batch_size,)

        log_prob1 = dist1.log_prob(action1)  # (batch_size,)

        dept1_embeddings = node_embeddings.gather(
            1, action1.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.node_hidden_dim)
        )  # (batch_size, 1, node_hidden_dim)

        dept1_embeddings_expanded = dept1_embeddings.expand(
            -1, num_nodes, -1
        )  # (batch_size, num_nodes, node_hidden_dim)
        combined_embeddings = torch.cat(
            [node_embeddings, dept1_embeddings_expanded], dim=-1
        )

        logits2: torch.Tensor = self.second_action_head(combined_embeddings).squeeze(-1) # (batch_size, num_nodes)
        
        mask2 = node_mask.clone()
        mask2.scatter_(1, action1.unsqueeze(-1), 0)

        dist2 = self._create_masked_distribution(logits2, mask2)

        if deterministic:
            action2 = dist2.probs.argmax(dim=-1)  #type: ignore (batch_size,)
        else:
            action2 = dist2.sample()  # (batch_size,)

        log_prob2 = dist2.log_prob(action2)  # (batch_size,)

        return action1, action2, log_prob1, log_prob2, dist1, dist2
    
    def get_log_prob_and_entropy(
            self,
            node_embeddings: torch.Tensor,
            node_mask: torch.Tensor,
            action1: torch.Tensor,
            action2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate selected action log_prob and entropy

        Args:
            node_embeddings (torch.Tensor): GNN node embeddings, shape (batch_size, num_nodes, node_hidden_dim)
            node_mask (torch.Tensor): Node mask, shape (batch_size, num_nodes) (1 for valid nodes, 0 for padding nodes)
            action1 (torch.Tensor): First selected action (node index), shape (batch_size,)
            action2 (torch.Tensor): Second selected action (node index), shape (batch_size,)

        Returns:
            log_prob1: torch.Tensor: Log probability of action1, shape (batch_size,)
            log_prob2: torch.Tensor: Log probability of action2, shape (batch_size,)
            total_log_prob: torch.Tensor: Sum of action1 and action2 log probability, shape (batch_size,)
            total_entropy: torch.Tensor: Sum of action1 and action2 entropy, shape (batch_size,)
        """

        batch_size, num_nodes, _ = node_embeddings.size()

        logits1 = self.first_action_head(node_embeddings).squeeze(-1)  # (batch_size, num_nodes)
        dist1 = self._create_masked_distribution(logits1, node_mask)
        log_prob1 = dist1.log_prob(action1) # (batch_size,)
        entropy1 = dist1.entropy() # (batch_size,)

        dept1_embeddings = node_embeddings.gather(
            1, action1.unsqueeze(-1).unsqueeze(-1).expand(batch_size, num_nodes, self.node_hidden_dim)
        )  # (batch_size, num_nodes, node_hidden_dim)
        combined_embeddings = torch.cat(
            [node_embeddings, dept1_embeddings], dim=-1
        )

        logits2 = self.second_action_head(combined_embeddings).squeeze(-1)  # (batch_size, num_nodes)

        mask2 = node_mask.clone()
        mask2.scatter_(1, action1.unsqueeze(-1), 0)

        dist2 = self._create_masked_distribution(logits2, mask2)
        log_prob2 = dist2.log_prob(action2) # (batch_size,)
        entropy2 = dist2.entropy() # (batch_size,)

        total_log_prob = log_prob1 + log_prob2
        total_entropy = entropy1 + entropy2

        return log_prob1, log_prob2, total_log_prob, total_entropy
    
    @torch.no_grad()
    def get_action(
            self,
            node_embeddings: torch.Tensor,
            node_mask: torch.Tensor,
            deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action only

        Args:
            node_embeddings (torch.Tensor): GNN node embeddings, shape (batch_size, num_nodes, node_hidden_dim)
            node_mask (torch.Tensor): Node mask, shape (batch_size, num_nodes) (1 for valid nodes, 0 for padding nodes)
            deterministic (bool, optional): Whether to select actions deterministically. Defaults to False.

        Returns:
            action1: torch.Tensor: First selected action (node index), shape (batch_size,)
            action2: torch.Tensor: Second selected action (node index), shape (batch_size,)
        """

        action1, action2, _, _, _, _ = self.forward(
            node_embeddings, node_mask, deterministic
        )

        return action1, action2