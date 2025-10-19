import torch
from typing import Any, Dict, Optional, Union
from tianshou.data import Batch
from tianshou.
from loguru import logger

from .ppo_model import LayoutOptimizationModel


class JointDist:
    def __init__(
        self,
        dist1: torch.distributions.Categorical,
        dist2: torch.distributions.Categorical,
    ):
        self.dist1 = dist1
        self.dist2 = dist2

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        action1 = actions[:, 0].long()
        action2 = actions[:, 1].long()
        log_prob1 = self.dist1.log_prob(action1)
        log_prob2 = self.dist2.log_prob(action2)
        return log_prob1 + log_prob2
    
    def entropy(self) -> torch.Tensor:
        return self.dist1.entropy() + self.dist2.entropy()
    
class LayoutPolicy(Pro)


# class LayoutPPOPolicy(PPOPolicy):
#     def __init__(
#         self,
#         model: LayoutOptimizationModel,
#         optim: torch.optim.Optimizer,
#         discount_factor: float = 0.99,
#         gae_lambda: float = 0.95,
#         vf_coef: float = 0.5,
#         ent_coef: float = 0.01,
#         max_grad_norm: float = 0.5,
#         value_clip: bool = True,
#         advantage_normalization: bool = True,
#         recompute_advantage: bool = False,
#         dual_clip: Optional[float] = None,
#         reward_normalization: bool = False,
#         eps_clip: float = 0.2,
#         **kwargs: Any,
#     ):
#         super().__init__(
#             actor=model,
#             critic=model,
#             optim=optim,
#             dist_fn=torch.distributions.Categorical,
#             discount_factor=discount_factor,
#             gae_lambda=gae_lambda,
#             vf_coef=vf_coef,
#             ent_coef=ent_coef,
#             max_grad_norm=max_grad_norm,
#             value_clip=value_clip,
#             advantage_normalization=advantage_normalization,
#             recompute_advantage=recompute_advantage,
#             dual_clip=dual_clip,
#             reward_normalization=reward_normalization,
#             **kwargs,
#         )

#         self._eps_clip = eps_clip
#         self._recompute_adv = recompute_advantage
#         self._adv_norm = advantage_normalization
#         self._weight_vf = vf_coef
#         self._weight_ent = ent_coef
#         self._grad_norm = max_grad_norm
#         self._dual_clip = dual_clip
#         self._value_clip = value_clip

#         self.model = model
#         self.logger = logger.bind(module=__name__)

#     def forward(
#         self,
#         batch: Batch,
#         state: Optional[Union[Dict, Batch, np.ndarray]] = None,
#         **kwargs: Any,
#     ) -> Batch:
#         node_embeddings, node_mask = self.model.encode_observations(batch.obs)

#         action1, action2, log_prob1, log_prob2, dist1, dist2 = self.model.actor(
#             node_embeddings=node_embeddings,
#             node_mask=node_mask,
#             deterministic=kwargs.get("deterministic", False),
#         )
#         actions = torch.stack([action1, action2], dim=-1)  # (batch_size, 2)
#         log_prob = log_prob1 + log_prob2  # (batch_size)

#         join_dist = JointDist(dist1, dist2)

#         value = self.model(batch.obs)

#         return Batch(
#             act=actions,
#             logp=log_prob,
#             state=state,
#             dist=join_dist,
#             value=value,
#         )

#     def learn(
#         self,
#         batch: Batch,
#         batch_size: Optional[int] = None,
#         repeat: int = 1,
#         **kwargs: Any,
#     ) -> Dict[Any, Any]:
#         losses = []

#         self.logger.info("Starting training loop")

#         total_samples = batch.obs.shape[0]

#         for _ in range(repeat):
#             indices = np.random.permutation(total_samples)

#             for start_idx in range(0, total_samples, batch_size or total_samples):
#                 end_idx = min(start_idx + (batch_size or total_samples), total_samples)
#                 mini_batch_indices = indices[start_idx:end_idx]
#                 mini_batch = batch[mini_batch_indices]

#                 log_prob, entropy = self.model.get_action_log_prob(
#                     mini_batch.obs,
#                     mini_batch.act,
#                 )

#                 value = self.model(mini_batch.obs)

#                 if self._recompute_adv:
#                     with torch.no_grad():
#                         value_old = self.model(mini_batch.obs)
#                         advantages = mini_batch.returns - value_old
#                 else:
#                     advantages = mini_batch.adv

#                 if self._adv_norm:
#                     advantages = (advantages - advantages.mean()) / (
#                         advantages.std() + 1e-8
#                     )

#                 ratio = (log_prob - mini_batch.logp_old).exp()
#                 surr1 = ratio * advantages
#                 surr2 = ratio.clamp(1 - self._eps_clip, 1 + self._eps_clip) * advantages

#                 if self._dual_clip:
#                     clip_loss = -torch.max(
#                         torch.min(surr1, surr2),
#                         self._dual_clip * advantages,
#                     ).mean()
#                 else:
#                     clip_loss = -torch.min(surr1, surr2).mean()

#                 if self._value_clip:
#                     value_clip = mini_batch.v_s + (value - mini_batch.v_s).clamp(
#                         -self._eps_clip, self._eps_clip
#                     )
#                     vf_loss = torch.max(
#                         (value - mini_batch.returns).pow(2),
#                         (value_clip - mini_batch.returns).pow(2),
#                     ).mean()
#                 else:
#                     vf_loss = (value - mini_batch.returns).pow(2).mean()

#                 ent_loss = -entropy.mean()

#                 loss = (
#                     clip_loss + self._weight_vf * vf_loss + self._weight_ent * ent_loss
#                 )

#                 self.optim.zero_grad()
#                 loss.backward()

#                 if self._grad_norm:
#                     torch.nn.utils.clip_grad_norm_(
#                         self.model.parameters(), self._grad_norm
#                     )

#                 self.optim.step()

#                 losses.append(
#                     {
#                         "loss": loss.item(),
#                         "loss/clip": clip_loss.item(),
#                         "loss/vf": vf_loss.item(),
#                         "loss/ent": ent_loss.item(),
#                     }
#                 )

#         avg_loss = {k: np.mean([loss[k] for loss in losses]) for k in losses[0].keys()}

#         return avg_loss
