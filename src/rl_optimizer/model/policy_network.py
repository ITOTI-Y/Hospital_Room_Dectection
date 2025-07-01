# src/rl_optimizer/model/policy_network.py

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict

from src.config import RLConfig
from src.rl_optimizer.utils.setup import setup_logger

logger = setup_logger(__name__)

class LayoutTransformer(BaseFeaturesExtractor):
    """
    基于Transformer的特征提取器，用于处理布局状态。

    该网络接收环境的字典式观测，通过嵌入层和Transformer编码器，
    学习布局中已放置科室与待放置科室之间的复杂空间和逻辑关系，
    最终为策略网络（Actor-Critic）输出一个固定维度的特征向量。

    该设计特别适用于自回归构建任务，因为它能有效地处理序列信息
    和元素之间的长程依赖。
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int, config: RLConfig):
        """
        初始化Transformer特征提取器。

        Args:
            observation_space (spaces.Dict): 环境的观测空间。
            features_dim (int): 输出特征的维度。
            config (RLConfig): RL优化器的配置对象。
        """
        super().__init__(observation_space, features_dim)

        self.config = config

        # 从观测空间中提取维度信息
        # 槽位数量，决定了序列长度
        num_slots = observation_space["layout"].shape[0]
        
        # 科室种类数量 (包括0，代表"未放置"或"空")
        # 支持两种观测空间定义方式
        if hasattr(observation_space["layout"], 'nvec'):
            # MultiDiscrete 空间
            num_depts = observation_space["layout"].nvec[0]
        else:
            # Box 空间
            num_depts = observation_space["layout"].high[0]

        embedding_dim = self.config.EMBEDDING_DIM

        # --- 网络层定义 ---
        
        # 1. 科室嵌入层 (Dept Embedding)
        # 将每个科室ID (包括0) 映射为一个高维向量
        self.dept_embedding = nn.Embedding(num_embeddings=num_depts, embedding_dim=embedding_dim)

        # 2. 槽位位置嵌入层 (Slot Positional Embedding)
        # 为每个槽位（位置）学习一个唯一的嵌入向量，以区分位置信息
        self.slot_position_embedding = nn.Embedding(num_embeddings=num_slots, embedding_dim=embedding_dim)


        # 3. 当前待决策科室嵌入层 (Current Dept Embedding)
        # 输入ID范围是 0 到 num_placeable_depts - 1
        self.current_dept_embedding = nn.Embedding(num_embeddings=num_depts, embedding_dim=embedding_dim)

        # 4. Transformer 编码器层
        # 这是网络的核心，用于处理序列信息
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=self.config.TRANSFORMER_HEADS,
            dim_feedforward=embedding_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True  # 确保输入张量的维度顺序为 (batch, sequence, features)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=self.config.TRANSFORMER_LAYERS
        )

        # 5. 输出线性层 (Output Layer)
        # 将处理后的特征与当前待决策科室的特征拼接，然后映射到最终的特征维度
        self.linear = nn.Sequential(
            nn.LayerNorm(embedding_dim * num_slots + embedding_dim),
            nn.Linear(embedding_dim * num_slots + embedding_dim, features_dim),
            nn.ReLU()
        )

        logger.info(f"LayoutTransformer 初始化成功。")
        logger.info(f"  - 槽位数量: {num_slots}")
        logger.info(f"  - 科室种类数量: {num_depts}")
        logger.info(f"  - 嵌入维度: {embedding_dim}")

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        定义模型的前向传播逻辑。

        Args:
            observations (Dict[str, torch.Tensor]): 从环境中获得的观测数据字典。
                - "layout": (batch_size, num_slots) 当前布局，值为科室ID
                - "current_dept_id": (batch_size, 1) 当前待决策的科室ID

        Returns:
            torch.Tensor: 提取出的状态特征，维度为 (batch_size, features_dim)。
        """
        # --- 1. 准备输入嵌入 ---
        layout_ids = observations["layout"]
        current_dept_id = observations["current_dept_id"].squeeze(-1)

        # 确保数据类型正确：嵌入层需要整数类型
        layout_ids = layout_ids.long()  # 转换为 LongTensor
        current_dept_id = current_dept_id.long()  # 转换为 LongTensor

        batch_size, num_slots = layout_ids.shape
        device = layout_ids.device

        # 获取科室嵌入
        dept_embeds = self.dept_embedding(layout_ids) # (B, num_slots, D_emb)

        # 获取槽位位置嵌入
        slot_positions = torch.arange(0, num_slots, device=device).unsqueeze(0).expand(batch_size, -1)
        slot_pos_embeds = self.slot_position_embedding(slot_positions) # (B, num_slots, D_emb)

        # 组合输入嵌入 (这是Transformer的标准做法)
        input_embeds = dept_embeds + slot_pos_embeds

        # --- 2. 通过 Transformer 处理布局信息 ---
        # Transformer的输出包含了每个槽位位置的上下文感知特征
        transformer_output = self.transformer_encoder(input_embeds) # (B, num_slots, D_emb)

        # 将所有槽位的特征展平，形成一个代表整个布局的向量
        flattened_layout_features = transformer_output.reshape(batch_size, -1) # (B, num_slots * D_emb)

        # --- 3. 整合当前待决策科室的信息 ---
        # 获取当前待决策科室的嵌入
        current_dept_embed = self.current_dept_embedding(current_dept_id) # (B, D_emb)

        # --- 4. 拼接并输出最终特征 ---
        # 将布局特征和当前科室特征拼接在一起
        combined_features = torch.cat([flattened_layout_features, current_dept_embed], dim=1)
        
        # 通过最后的线性层得到最终的特征向量
        final_features = self.linear(combined_features)
        
        return final_features