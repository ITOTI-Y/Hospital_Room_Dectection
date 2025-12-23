import torch
import torch.nn as nn
from torch.distributions import Categorical


class AutoregressiveActor(nn.Module):
    def __init__(
        self,
        node_hidden_dim: int,
        actor_hidden_dim: int,
        dropout: float = 0.1,
        eval_temperature: float = 0.3,
    ):
        """
        Autoregression Policy Network, used to select two departments to swap.

        Args:
            node_hidden_dim (int): GNN output dimension
            actor_hidden_dim (int): Actor MLP hidden dimension
            dropout (float, optional): Dropout rate
            eval_temperature (float, optional): Temperature for evaluation sampling (lower = more deterministic)
        """

        super().__init__()

        self.node_hidden_dim: int = node_hidden_dim
        self.actor_hidden_dim: int = actor_hidden_dim
        self.eval_temperature: float = eval_temperature

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
        self, logits: torch.Tensor, mask: torch.Tensor
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
        action_history_mask: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Categorical | None,
        Categorical | None,
    ]:
        """
        Forward, execute autoregressive two action select

        Args:
            node_embeddings (torch.Tensor): GNN node embeddings, shape (batch_size, num_nodes, node_hidden_dim)
            node_mask (torch.Tensor): Node mask, shape (batch_size, num_nodes) (1 for valid nodes, 0 for padding nodes)
            deterministic (bool): Whether to use deterministic action selection
            action_history_mask (torch.Tensor, optional): Mask for action pairs to avoid,
                shape (batch_size, num_nodes, num_nodes). 0 means avoid, 1 means allowed.

        Returns:
            action1: torch.Tensor: First selected action (node index), shape (batch_size,)
            action2: torch.Tensor: Second selected action (node index), shape (batch_size,)
            log_prob1: torch.Tensor: Log probability of action1, shape (batch_size,)
            log_prob2: torch.Tensor: Log probability of action2, shape (batch_size,)
            dist1: Optional[Categorical]: Action distribution for action1, None if deterministic is True
            dist2: Optional[Categorical]: Action distribution for action2, None if deterministic is
        """

        _, num_nodes, _ = node_embeddings.size()

        logits1: torch.Tensor = self.first_action_head(node_embeddings).squeeze(
            -1
        )  # (batch_size, num_nodes)

        # Apply action history mask to first action if provided
        if action_history_mask is not None:
            # For first action, penalize nodes that have many blocked pairs
            # Count how many destinations are blocked for each source node
            blocked_count = (1 - action_history_mask).sum(dim=-1)  # (batch_size, num_nodes)
            # Soft penalty: reduce logits for nodes with many blocked pairs
            logits1 = logits1 - blocked_count * 0.5

        dist1 = self._create_masked_distribution(logits1, node_mask)

        # Use temperature-based sampling instead of pure argmax for deterministic mode
        if deterministic:
            # Apply temperature scaling for softer decisions
            scaled_logits1 = logits1 / self.eval_temperature
            temp_dist1 = self._create_masked_distribution(scaled_logits1, node_mask)
            action1 = temp_dist1.sample()
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

        logits2: torch.Tensor = self.second_action_head(combined_embeddings).squeeze(
            -1
        )  # (batch_size, num_nodes)

        mask2 = node_mask.clone()
        mask2.scatter_(1, action1.unsqueeze(-1), 0)

        # Apply action history mask for second action based on selected first action
        if action_history_mask is not None:
            # Get the row corresponding to action1 from the mask
            batch_indices = torch.arange(action1.size(0), device=action1.device)
            pair_mask = action_history_mask[batch_indices, action1]  # (batch_size, num_nodes)
            # Combine with existing mask
            mask2 = mask2 * pair_mask.long()

        dist2 = self._create_masked_distribution(logits2, mask2)

        # Use temperature-based sampling for deterministic mode
        if deterministic:
            scaled_logits2 = logits2 / self.eval_temperature
            temp_dist2 = self._create_masked_distribution(scaled_logits2, mask2)
            action2 = temp_dist2.sample()
        else:
            action2 = dist2.sample()  # (batch_size,)

        log_prob2 = dist2.log_prob(action2)  # (batch_size,)

        return action1, action2, log_prob1, log_prob2, dist1, dist2

    @torch.no_grad()
    def get_action(
        self,
        node_embeddings: torch.Tensor,
        node_mask: torch.Tensor,
        deterministic: bool = False,
        action_history_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get action only

        Args:
            node_embeddings (torch.Tensor): GNN node embeddings, shape (batch_size, num_nodes, node_hidden_dim)
            node_mask (torch.Tensor): Node mask, shape (batch_size, num_nodes) (1 for valid nodes, 0 for padding nodes)
            deterministic (bool, optional): Whether to select actions deterministically. Defaults to False.
            action_history_mask (torch.Tensor, optional): Mask for action pairs to avoid

        Returns:
            action1: torch.Tensor: First selected action (node index), shape (batch_size,)
            action2: torch.Tensor: Second selected action (node index), shape (batch_size,)
        """

        action1, action2, _, _, _, _ = self.forward(
            node_embeddings, node_mask, deterministic, action_history_mask
        )

        return action1, action2
