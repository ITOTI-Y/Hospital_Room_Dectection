# -*- coding: utf-8 -*-
"""成本计算引擎和程序化就医流程生成器。

该模块负责根据元规则动态生成多样的就医流程，并为强化学习环境提供
高效的成本计算方法。
"""
import random
import yaml
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from loguru import logger

from src.config_loader import settings


class PathwayGenerator:
    """根据元规则程序化生成就医流程。

    该类解析 `pathways.yaml` 中的 `training_generation` 配置，
    并在每次调用 `generate` 方法时，创建一个新的、随机但符合逻辑的就医流程。
    """

    def __init__(self):
        """初始化 PathwayGenerator。"""
        if not settings:
            raise RuntimeError("配置尚未加载，无法初始化 PathwayGenerator。")
        
        self.config = settings.pathways.training_generation
        self.pools = self.config.get("department_pools", {})
        self.fragments = self.config.get("sequence_fragments", {})
        self.meta_rules = self.config.get("meta_rules", [])
        logger.info("PathwayGenerator 初始化成功。")

    def generate(self) -> Dict[str, Any]:
        """生成一个随机的就医流程。

        Returns:
            一个字典，代表一个生成的就医流程，包含ID、描述、序列和权重。
        """
        rule_weights = [rule.get("base_weight", 1.0) for rule in self.meta_rules]
        chosen_rule = random.choices(self.meta_rules, weights=rule_weights, k=1)[0]

        generated_sequence: List[str] = []
        context: Dict[str, str] = {}

        for step in chosen_rule.get("structure", []):
            self._parse_step(step, context, generated_sequence)

        final_pathway = {
            "process_id": f"PROC_GEN_{chosen_rule['id']}_{random.randint(1000, 9999)}",
            "description": f"程序化生成: {chosen_rule['description']}",
            "core_sequence": generated_sequence,
            "start_nodes": chosen_rule.get("start_nodes", ["门"]),
            "end_nodes": chosen_rule.get("end_nodes", ["门"]),
            "base_weight": chosen_rule.get("base_weight", 1.0),
        }
        
        full_sequence_str = (
            f"{final_pathway['start_nodes']} -> "
            f"{' -> '.join(generated_sequence)} -> "
            f"{final_pathway['end_nodes']}"
        )
        logger.debug(
            f"生成流程: {final_pathway['process_id']}, 完整序列: {full_sequence_str}"
        )
        return final_pathway

    def _parse_step(
        self,
        step: Dict[str, Any],
        context: Dict[str, str],
        generated_sequence: List[str],
    ):
        """递归解析单个步骤并更新序列。"""
        step_type = step.get("type")

        if step_type == "primary_department":
            from_pool_value = step.get("from_pool")
            department_list = []
            if isinstance(from_pool_value, str) and from_pool_value in self.pools:
                department_list = self.pools[from_pool_value]
            elif isinstance(from_pool_value, list):
                department_list = from_pool_value
            
            if department_list:
                dept = random.choice(department_list)
                context["primary_department"] = dept
                generated_sequence.append(dept)

        elif step_type == "fixed":
            dept_name = step.get("name")
            if dept_name:
                generated_sequence.append(dept_name)

        elif step_type == "pool":
            pool_name = step.get("pool_name")
            if pool_name in self.pools:
                generated_sequence.append(random.choice(self.pools[pool_name]))

        elif step_type == "optional":
            if random.random() < step.get("probability", 1.0):
                self._parse_step(step.get("step", {}), context, generated_sequence)

        elif step_type == "fragment":
            fragment_name = step.get("fragment_name")
            if fragment_name in self.fragments:
                fragment_steps = self.fragments[fragment_name]
                for frag_step in fragment_steps:
                    self._parse_step(frag_step, context, generated_sequence)

        elif step_type == "repeat":
            target = step.get("target")
            if target in context:
                generated_sequence.append(context[target])


class CostEngine:
    """成本计算引擎。

    在训练阶段，使用 PathwayGenerator 动态生成流程。
    在评估阶段，加载固定的流程文件。
    提供高效的增量成本计算方法。
    """

    def __init__(
        self,
        travel_times_df: pd.DataFrame,
        service_times: Dict[str, float],
        is_training: bool = True,
        num_pathways_for_training: int = 200,
    ):
        """初始化 CostEngine。

        Args:
            travel_times_df: 包含所有槽位之间通行时间的DataFrame。
            service_times: 科室名称到其服务时间的映射。
            is_training: 是否为训练模式。True则使用程序化生成，False则加载固定场景。
            num_pathways_for_training: 在训练模式下，用于预计算权重矩阵的流程数量。
        """
        self.is_training = is_training
        self.travel_times = travel_times_df
        self.service_times = service_times
        self.pathways: List[Dict[str, Any]] = []
        self.pair_weights: Dict[Tuple[str, str], float] = defaultdict(float)
        self.service_weights: Dict[str, float] = defaultdict(float)
        self.related_depts: Dict[str, List[str]] = defaultdict(list)
        self._cost_cache: Dict[frozenset, float] = {}

        if self.is_training:
            self.generator = PathwayGenerator()
            self.pathways = self._get_training_pathways(num_pathways_for_training)
        else:
            # TODO(Roo): 实现加载评估场景的逻辑
            self.pathways = []

        self._precompute_weights()

    def _get_training_pathways(self, num_pathways: int) -> List[Dict[str, Any]]:
        """使用生成器创建一批用于训练的就医流程。"""
        logger.info(f"正在生成 {num_pathways} 个训练流程...")
        return [self.generator.generate() for _ in range(num_pathways)]

    def _precompute_weights(self):
        """
        预计算科室对的加权共现频率（用于通行成本）和单个科室的加权出现频率（用于服务成本）。
        """
        logger.info("正在预计算通行和服务权重...")
        if not self.pathways:
            logger.warning("流程列表为空，无法计算权重。")
            return

        for pathway in self.pathways:
            sequence = pathway.get("core_sequence", [])
            if not sequence:
                continue

            weight = pathway.get("base_weight", 1.0)
            start_nodes = pathway.get("start_nodes", [])
            end_nodes = pathway.get("end_nodes", [])
            
            for dept in sequence:
                self.service_weights[dept] += weight

            if start_nodes:
                weight_per_start = weight / len(start_nodes)
                for start_node in start_nodes:
                    pair = (start_node, sequence[0]) if start_node < sequence[0] else (sequence[0], start_node)
                    self.pair_weights[pair] += weight_per_start

            for i in range(len(sequence) - 1):
                dept1, dept2 = sequence[i], sequence[i+1]
                pair = (dept1, dept2) if dept1 < dept2 else (dept2, dept1)
                self.pair_weights[pair] += weight

            if end_nodes:
                weight_per_end = weight / len(end_nodes)
                for end_node in end_nodes:
                    pair = (sequence[-1], end_node) if sequence[-1] < end_node else (end_node, sequence[-1])
                    self.pair_weights[pair] += weight_per_end
        
        for dept1, dept2 in self.pair_weights.keys():
            self.related_depts[dept1].append(dept2)
            self.related_depts[dept2].append(dept1)

        logger.info(
            f"✅ 权重预计算完成，共 {len(self.pair_weights)} 个科室对, "
            f"{len(self.service_weights)} 个科室服务。"
        )

    def calculate_total_cost(self, layout: Dict[str, str]) -> float:
        """计算给定布局的总成本（通行时间 + 服务时间），使用缓存。"""
        layout_key = frozenset(layout.items())
        if layout_key in self._cost_cache:
            return self._cost_cache[layout_key]

        travel_cost = 0.0
        for (dept1, dept2), weight in self.pair_weights.items():
            slot1 = layout.get(dept1)
            slot2 = layout.get(dept2)
            if slot1 and slot2:
                travel_cost += self.travel_times.at[slot1, slot2] * weight
        
        service_cost = 0.0
        for dept, weight in self.service_weights.items():
            service_time = self.service_times.get(dept, 0)
            service_cost += service_time * weight
            
        total_cost = travel_cost + service_cost
        self._cost_cache[layout_key] = total_cost
        return total_cost

    def calculate_swap_delta_cost(
        self,
        dept_a: str,
        dept_b: str,
        slot_a: str,
        slot_b: str,
        layout: Dict[str, str],
    ) -> float:
        """
        高效计算交换两个科室后的总通行成本变化量 (Delta)。
        """
        cost_before = 0.0
        cost_after = 0.0

        for other_dept in self.related_depts.get(dept_a, []):
            if other_dept == dept_b: continue
            pair = (dept_a, other_dept) if dept_a < other_dept else (other_dept, dept_a)
            weight = self.pair_weights.get(pair, 0)
            if weight == 0: continue
            slot_other = layout.get(other_dept)
            if not slot_other: continue
            cost_before += self.travel_times.at[slot_a, slot_other] * weight
            cost_after += self.travel_times.at[slot_b, slot_other] * weight

        for other_dept in self.related_depts.get(dept_b, []):
            if other_dept == dept_a: continue
            pair = (dept_b, other_dept) if dept_b < other_dept else (other_dept, dept_b)
            weight = self.pair_weights.get(pair, 0)
            if weight == 0: continue
            slot_other = layout.get(other_dept)
            if not slot_other: continue
            cost_before += self.travel_times.at[slot_b, slot_other] * weight
            cost_after += self.travel_times.at[slot_a, slot_other] * weight
            
        pair_ab = (dept_a, dept_b) if dept_a < dept_b else (dept_b, dept_a)
        weight_ab = self.pair_weights.get(pair_ab, 0)
        if weight_ab > 0:
            cost_before += self.travel_times.at[slot_a, slot_b] * weight_ab
            cost_after += self.travel_times.at[slot_b, slot_a] * weight_ab

        return cost_after - cost_before


# 示例用法
if __name__ == "__main__":
    logger.info("测试 CostEngine...")
    if not settings or not settings.paths:
        logger.error("配置或路径配置加载失败，无法执行测试。")
        exit(1)

    try:
        # 1. 加载真实数据
        travel_times_path = Path(settings.paths.travel_times_csv)
        slots_path = Path(settings.paths.slots_csv)
        graph_config_path = Path("configs/graph_config.yaml") # 这个由config_loader处理，暂时保留硬编码
        
        if not travel_times_path.exists() or not graph_config_path.exists() or not slots_path.exists():
            logger.error("测试所需的数据或配置文件不存在，无法执行。")
            exit(1)
            
        travel_times_df = pd.read_csv(travel_times_path, index_col=0)
        with open(graph_config_path, "r", encoding="utf-8") as f:
            graph_config = yaml.safe_load(f)

        # 2. 解析科室数据
        node_defs = graph_config.get("node_definitions", {})
        service_times = {name: data.get("service_time", 0) for name, data in node_defs.items()}
        
        # 3. 从 slots.csv 创建初始布局
        logger.info(f"从 '{slots_path}' 加载初始布局...")
        slots_df = pd.read_csv(slots_path)
        initial_layout = {
            row['name']: str(row['id'])
            for _, row in slots_df.iterrows()
            if row['category'] == 'SLOT'
        }
        depts_to_layout = list(initial_layout.keys())
        logger.info(f"成功加载 {len(initial_layout)} 个科室的初始布局。")

        # 4. 初始化并测试 CostEngine
        engine = CostEngine(
            travel_times_df=travel_times_df,
            service_times=service_times,
            is_training=True
        )
        
        sorted_weights = sorted(engine.pair_weights.items(), key=lambda item: item[1], reverse=True)
        logger.info(f"计算出的科室对权重 (Top 5): {sorted_weights[:5]}")

        total_cost = engine.calculate_total_cost(initial_layout)
        logger.info(f"初始布局的总成本: {total_cost:,.2f}")
        total_cost_cached = engine.calculate_total_cost(initial_layout)
        logger.info(f"缓存后的总成本: {total_cost_cached:,.2f} (应与上面相同)")

        # 5. 测试增量计算
        if len(depts_to_layout) >= 2:
            dept1, dept2 = random.sample(depts_to_layout, 2)
            slot1, slot2 = initial_layout.get(dept1), initial_layout.get(dept2)
            
            if slot1 and slot2:
                delta = engine.calculate_swap_delta_cost(dept1, dept2, slot1, slot2, initial_layout)
                logger.info(f"随机交换 '{dept1}' 和 '{dept2}' 的成本变化 (Delta): {delta:.2f}")
        else:
            logger.warning("可交换科室数量不足，无法测试交换。")

    except Exception as e:
        logger.exception(f"测试失败: {e}")
