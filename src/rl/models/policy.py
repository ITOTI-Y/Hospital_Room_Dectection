from typing import Any

import numpy as np
import torch
from gymnasium import Space
from loguru import logger
from tianshou.data import Batch, ReplayBuffer, to_torch_as
from tianshou.data.stats import SequenceSummaryStats
from tianshou.policy import A2CPolicy
from tianshou.policy.modelfree.a2c import A2CTrainingStats

from .ppo_model import LayoutOptimizationModel


class LayoutPolicy(A2CPolicy):
    def __init__(
        self,
        model: LayoutOptimizationModel,
        optim: torch.optim.Optimizer,
        action_space: Space,
        discount_factor: float,
        gae_lambda: float,
        vf_coef: float,
        ent_coef: float,
        max_grad_norm: float,
        value_clip: bool,
        advantage_normalization: bool,
        recompute_advantage: bool,
        dual_clip: float | None,
        reward_normalization: bool,
        eps_clip: float,
        max_batchsize: int,
        deterministic_eval: bool,
        **kwargs: Any,
    ):
        def dummy_dist_fn(logits: torch.Tensor):
            return torch.distributions.Categorical(logits=logits)

        super().__init__(
            actor=model,
            critic=model,
            optim=optim,
            dist_fn=dummy_dist_fn,
            action_space=action_space,
            discount_factor=discount_factor,
            gae_lambda=gae_lambda,
            vf_coef=vf_coef,
            ent_coef=ent_coef,
            max_grad_norm=max_grad_norm,
            reward_normalization=reward_normalization,
            max_batchsize=max_batchsize,
            deterministic_eval=deterministic_eval,
            action_scaling=False,  # MultiDiscrete does not need action scaling
            **kwargs,
        )

        self._eps_clip = eps_clip
        self._value_clip = value_clip
        self._adv_norm = advantage_normalization
        self._recompute_adv = recompute_advantage
        self._dual_clip = dual_clip
        self._vf_coef = vf_coef
        self._ent_coef = ent_coef
        self._grad_norm = max_grad_norm
        self._max_batchsize = max_batchsize

        self.model = model
        self.logger = logger.bind(module=__name__)

    def forward(
        self,
        batch: Batch,
        state: dict | Batch | torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Batch:
        node_embeddings, node_mask = self.model.encode_observations(batch.obs)

        deterministic = kwargs.get("deterministic", False)
        if self.deterministic_eval and not self.training:
            deterministic = True

        action1, action2, log_prob1, log_prob2, dist1, dist2 = self.model.actor(
            node_embeddings=node_embeddings,
            node_mask=node_mask,
            deterministic=deterministic,
        )

        value = self.model.critic(
            node_embeddings=node_embeddings,
            node_mask=node_mask,
        ).flatten()  # (batch_size,)

        actions = torch.stack([action1, action2], dim=-1)  # (batch_size, 2)

        joint_logp = log_prob1 + log_prob2  # (batch_size,)
        joint_entropy = dist1.entropy() + dist2.entropy()

        return Batch(
            act=actions,
            state=state,
            logp=joint_logp,
            entropy=joint_entropy,
            value=value,
        )

    def process_fn(
        self,
        batch: Batch,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ):
        if self._recompute_adv:
            self._buffer, self._indices = buffer, indices

        batch = self._compute_returns(batch, buffer, indices)  # type: ignore

        batch.act = to_torch_as(batch.act, batch.v_s)

        old_log_prob = []
        with torch.no_grad():
            for minibatch in batch.split(
                self._max_batchsize, shuffle=False, merge_last=True
            ):  # type: ignore
                result = self(minibatch)
                old_log_prob.append(result.logp)

        batch.logp_old = torch.cat(old_log_prob, dim=0)

        return batch

    def learn(
        self,
        batch: Batch,
        batch_size: int,
        repeat: int,
    ):
        losses, clip_losses, vf_losses, ent_losses = [], [], [], []

        for step in range(repeat):
            if self._recompute_adv and step > 0:
                batch = self._compute_returns(batch, self._buffer, self._indices)  # type: ignore

            for minibatch in batch.split(batch_size, shuffle=True, merge_last=True):
                result = self(minibatch)

                if self._adv_norm:
                    mean, std = minibatch.adv.mean(), minibatch.adv.std()
                    minibatch.adv = (minibatch.adv - mean) / (std + 1e-8)

                log_prob = result.logp
                ratio = (log_prob - minibatch.logp_old).exp().float()

                ratio = ratio  # (batch_size,)
                surr1 = ratio * minibatch.adv
                surr2 = (
                    ratio.clamp(1.0 - self._eps_clip, 1.0 + self._eps_clip)
                    * minibatch.adv
                )

                if self._dual_clip:
                    clip1 = torch.min(surr1, surr2)
                    clip2 = torch.max(clip1, self._dual_clip * minibatch.adv)
                    clip_loss = -torch.where(minibatch.adv < 0, clip2, clip1).mean()
                else:
                    clip_loss = -torch.min(surr1, surr2).mean()

                value = result.value

                if self._value_clip:
                    v_clip = minibatch.v_s + (value - minibatch.v_s).clamp(
                        -self._eps_clip, self._eps_clip
                    )
                    vf1 = (minibatch.returns - value).pow(2)
                    vf2 = (minibatch.returns - v_clip).pow(2)
                    vf_loss = torch.max(vf1, vf2).mean()
                else:
                    vf_loss = (minibatch.returns - value).pow(2).mean()

                ent_loss = -result.entropy.mean()

                loss = clip_loss + self._vf_coef * vf_loss + self._ent_coef * ent_loss

                self.optim.zero_grad()
                loss.backward()

                if self._grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self._grad_norm,  # type: ignore
                    )

                self.optim.step()

                clip_losses.append(clip_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(ent_loss.item())
                losses.append(loss.item())

        return A2CTrainingStats(
            loss=SequenceSummaryStats.from_sequence(losses),
            actor_loss=SequenceSummaryStats.from_sequence(clip_losses),
            vf_loss=SequenceSummaryStats.from_sequence(vf_losses),
            ent_loss=SequenceSummaryStats.from_sequence(ent_losses),
        )
