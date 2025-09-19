# -*- coding: utf-8 -*-
"""配置加载器模块。

使用 Pydantic V2 和 PyYAML 安全地加载、验证和合并来自多个YAML文件的配置。
提供一个全局单例 `settings` 对象，以便在项目的任何地方轻松访问配置。

遵循 Google Python Style Guide。
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Annotated

import yaml
from loguru import logger
from pydantic import BaseModel, Field, model_validator

# ===================================================================
# Pydantic 模型定义 (与 YAML 文件结构对应)
# ===================================================================


class PathsConfig(BaseModel, extra='forbid'):
    """项目路径配置。"""

    data_dir: str
    label_dir: str
    results_dir: str
    network_dir: str
    model_dir: str
    debug_dir: str
    travel_times_csv: str
    slots_csv: str

    @model_validator(mode='after')
    def interpolate_paths(self) -> 'PathsConfig':
        """使用模型中已有的值进行路径插值。"""
        # 使用 __dict__ 避免 Pydantic 的私有属性警告
        path_vars = self.__dict__
        for field_name, value in self:
            if isinstance(value, str):
                # 使用 format_map 进行安全的插值
                interpolated_value = value.format_map(path_vars)
                setattr(self, field_name, interpolated_value)
        return self


class AgentConfig(BaseModel, extra='forbid'):
    """DRL Agent (e.g., PPO) 的配置。

    Attributes:
        learning_rate: 优化器的学习率。
        gamma: 强化学习中的折扣因子。
    """

    learning_rate: float = Field(..., description="优化器的学习率。")
    gamma: float = Field(..., description="强化学习中的折扣因子。")
    # TODO(Roo): 后续根据选择的Tianshou Agent补充更多特定参数。


class AdjacencyPreference(BaseModel, extra='forbid'):
    """定义两个科室之间的邻接偏好。

    Attributes:
        depts: 需要相邻的两个科室的名称列表。
        weight: 此邻接偏好的重要性权重，用于计算奖励。
    """

    depts: Annotated[
        List[str],
        Field(min_length=2, max_length=2, description="需要相邻的两个科室的名称列表。"),
    ]
    weight: float = Field(
        1.0, gt=0, description="此邻接偏好的重要性权重，用于计算奖励。"
    )


class ConstraintsConfig(BaseModel, extra='forbid'):
    """约束相关的配置。

    Attributes:
        area_compatibility_tolerance: 科室与槽位面积的兼容性容忍度 (0.0-1.0)。
            例如，0.1代表科室面积不能超过槽位面积的110%，也不能低于90%。
        fixed_departments: 在布局优化中位置固定的科室列表。
        adjacency_preferences: 软约束，定义了期望相邻的科室对及其权重。
        hard_constraint_penalty: 违反硬约束（如面积不兼容）时施加的巨大负奖励。
        adjacency_reward_factor: 满足邻接偏好时，用于放大基础奖励的系数。
    """

    area_compatibility_tolerance: float = Field(
        ...,
        gt=0,
        lt=1,
        description="科室与槽位面积的兼容性容忍度 (0.0-1.0)。",
    )
    fixed_departments: List[str] = Field(
        [], description="在布局优化中位置固定的科室列表。"
    )
    adjacency_preferences: List[AdjacencyPreference] = Field(
        [], description="软约束，定义了期望相邻的科室对及其权重。"
    )
    hard_constraint_penalty: float = Field(
        ..., description="违反硬约束时施加的巨大负奖励。"
    )
    adjacency_reward_factor: float = Field(
        ..., gt=0, description="满足邻接偏好时，用于放大基础奖励的系数。"
    )


class PathwaysConfig(BaseModel, extra='forbid'):
    """就医流程 (Pathways) 相关的配置。

    Attributes:
        training_generation: 用于训练阶段的程序化流程生成元规则。
        evaluation_scenarios: 用于评估阶段的、指向固定流程文件的场景定义。
    """

    training_generation: Dict[str, Any] = Field(
        ..., description="用于训练阶段的程序化流程生成元规则。"
    )
    evaluation_scenarios: Dict[str, Any] = Field(
        ..., description="用于评估阶段的、指向固定流程文件的场景定义。"
    )


class GlobalSettings(BaseModel, extra='forbid'):
    """主配置模型，聚合所有子配置。

    Attributes:
        agent: Agent相关的配置。
        constraints: 约束相关的配置。
        pathways: 就医流程相关的配置。
    """

    paths: PathsConfig
    agent: AgentConfig
    constraints: ConstraintsConfig
    pathways: PathwaysConfig


class ConfigLoader:
    """负责加载、合并和验证配置的类。

    Attributes:
        config_dir: 存放YAML配置文件的目录路径。
    """

    def __init__(self, config_dir: Path = Path("configs")):
        """初始化ConfigLoader。

        Args:
            config_dir: 存放YAML配置文件的目录路径。
        """
        self.config_dir = config_dir
        self._settings = self._load_and_validate()

    def _load_yaml(self, file_path: Path) -> Dict[str, Any]:
        """加载单个YAML文件。

        Args:
            file_path: YAML文件的路径。

        Returns:
            从YAML文件加载的字典内容。

        Raises:
            FileNotFoundError: 如果指定的配置文件不存在。
        """
        if not file_path.exists():
            logger.error(f"配置文件不存在: {file_path}")
            raise FileNotFoundError(f"配置文件不存在: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _load_and_validate(self) -> GlobalSettings:
        """加载所有YAML文件并使用Pydantic进行验证。

        Returns:
            一个经过验证的、包含所有配置的GlobalSettings实例。

        Raises:
            pydantic.ValidationError: 如果配置内容不符合模型定义。
        """
        paths_raw = self._load_yaml(self.config_dir / "paths.yaml")
        agent_raw = self._load_yaml(self.config_dir / "agent.yaml")
        constraints_raw = self._load_yaml(self.config_dir / "constraints.yaml")
        pathways_raw = self._load_yaml(self.config_dir / "pathways.yaml")

        full_config_raw = {
            "paths": paths_raw,
            "agent": agent_raw,
            "constraints": constraints_raw,
            "pathways": pathways_raw,
        }

        validated_config = GlobalSettings.model_validate(full_config_raw)
        logger.info("✅ 配置加载并验证成功!")
        return validated_config

    @property
    def settings(self) -> GlobalSettings:
        """提供对已验证配置的只读访问。"""
        return self._settings


# ===================================================================
# 全局单例 (Singleton)
# ===================================================================
# 创建一个全局实例，以便在整个应用程序中通过导入来访问。
#
# 用法:
# from src.config_loader import settings
# learning_rate = settings.agent.learning_rate
#
settings: Optional[GlobalSettings] = None
try:
    loader = ConfigLoader()
    settings = loader.settings
except Exception as e:
    logger.critical(f"❌ 加载配置时发生严重错误: {e}")
    # 在CI/CD或某些测试环境中，配置文件可能不存在，允许settings为None。
    settings = None
