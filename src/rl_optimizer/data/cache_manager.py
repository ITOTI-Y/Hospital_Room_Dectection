import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import itertools

from src.config import RLConfig
from src.rl_optimizer.utils.setup import setup_logger, save_json, load_json, save_pickle, load_pickle

logger = setup_logger(__name__)

class CacheManager:

    def __init__(self, config: RLConfig):
        """
        初始化缓存管理器，并处理所有数据的加载和预计算。

        Args:
            config (RLConfig): RL优化器的配置对象。
        """
        self.config = config

        # --- 1. 加载和处理基础数据 ---
        self.raw_travel_times_df = self._load_raw_travel_times()
        self.node_data = self._load_and_process_node_data(self.raw_travel_times_df)
        self.travel_times_matrix = self.raw_travel_times_df.drop('面积', errors='ignore')

        # --- 2. 节点分类与封装 ---
        # 核心逻辑：所有节点分类都在此处完成并封装
        fixed_mask = self.node_data['generic_name'].isin(config.FIXED_NODE_TYPES)
        self.all_nodes_list: List[str] = self.node_data['node_id'].tolist()
        self.placeable_nodes_df: pd.DataFrame = self.node_data[~fixed_mask].copy()
        self.fixed_nodes_df: pd.DataFrame = self.node_data[fixed_mask].copy()

        # 公开给外部使用的、清晰的属性
        self.placeable_departments: List[str] = self.placeable_nodes_df['node_id'].tolist()
        self.placeable_slots: List[str] = self.placeable_nodes_df['node_id'].tolist()

        # --- 3. 解析流程与流线 ---
        self.variants = self.get_node_variants()
        self.traffic = self.get_traffic_distribution()
        self.resolved_pathways = self.get_resolved_pathways()

    def _load_raw_travel_times(self) -> pd.DataFrame:
        """加载原始CSV文件。"""
        if not self.config.TRAVEL_TIMES_CSV.exists():
            logger.error(f"致命错误: 原始通行时间文件未找到: {self.config.TRAVEL_TIMES_CSV}")
            raise FileNotFoundError(f"原始通行时间文件未找到: {self.config.TRAVEL_TIMES_CSV}")
        return pd.read_csv(self.config.TRAVEL_TIMES_CSV, index_col=0)

    def _load_and_process_node_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """从已加载的DataFrame中解析节点和面积。"""
        logger.info("正在处理节点数据...")
        node_ids = df.columns.tolist()
        
        if '面积' not in df.index:
            raise ValueError("CSV文件中必须包含名为'面积'的最后一行。")
        area_series = df.loc['面积']

        if not all(node_id in area_series.index for node_id in node_ids):
             raise ValueError("CSV文件的列标题与'面积'行的索引不完全匹配。请检查文件格式。")

        nodes_list = [{'node_id': node_id, 'area': area_series[node_id]} for node_id in node_ids]
        all_nodes_df = pd.DataFrame(nodes_list)
        all_nodes_df['generic_name'] = all_nodes_df['node_id'].apply(lambda x: str(x).split('_')[0])
        
        logger.info(f"成功处理了 {len(all_nodes_df)} 个唯一节点及其面积。")
        return all_nodes_df

    def get_node_variants(self) -> Dict[str, List[str]]:
        """
        获取节点变体映射。如果缓存存在则加载，否则从节点数据生成。
        """
        if self.config.NODE_VARIANTS_JSON.exists():
            logger.info(f"从缓存加载节点变体: {self.config.NODE_VARIANTS_JSON}")
            return load_json(self.config.NODE_VARIANTS_JSON)
        
        logger.info("缓存未找到，正在生成节点变体...")
        variants = self.node_data.groupby('generic_name')['node_id'].apply(list).to_dict()
        save_json(variants, self.config.NODE_VARIANTS_JSON)
        logger.info(f"节点变体已生成并缓存至: {self.config.NODE_VARIANTS_JSON}")
        return variants
    
    def get_traffic_distribution(self) -> Dict[str, Dict[str, float]]:
        """
        获取流量分布。如果缓存存在则加载，否则基于变体生成初始权重。
        """
        if self.config.TRAFFIC_DISTRIBUTION_JSON.exists():
            logger.info(f"从缓存加载流量分布: {self.config.TRAFFIC_DISTRIBUTION_JSON}")
            return load_json(self.config.TRAFFIC_DISTRIBUTION_JSON)
            
        logger.info("缓存未找到，正在生成初始流量分布...")
        traffic_distribution = {
            generic_name: {node: 1.0 for node in specific_nodes}
            for generic_name, specific_nodes in self.variants.items()
        }
        save_json(traffic_distribution, self.config.TRAFFIC_DISTRIBUTION_JSON)
        logger.info(f"初始流量分布已生成并缓存至: {self.config.TRAFFIC_DISTRIBUTION_JSON}")
        return traffic_distribution
    
    def get_resolved_pathways(self) -> List[Tuple[List[str], float]]:
        """
        获取最终解析出的流线列表。这是调度核心，如果缓存不存在则触发解析。
        """
        if self.config.RESOLVED_PATHWAYS_PKL.exists():
            logger.info(f"从缓存加载已解析的流线: {self.config.RESOLVED_PATHWAYS_PKL}")
            return load_pickle(self.config.RESOLVED_PATHWAYS_PKL)

        logger.info("缓存未找到，正在解析就医流程以生成具体流线...")
        
        try:
            templates = load_json(self.config.PROCESS_TEMPLATES_JSON)
        except FileNotFoundError:
            logger.error(f"致命错误: 就医流程模板文件未找到: {self.config.PROCESS_TEMPLATES_JSON}")
            raise
        
        resolved_pathways = []
        for template in templates:
            resolved_pathways.extend(self._resolve_single_template(template))
        
        save_pickle(resolved_pathways, self.config.RESOLVED_PATHWAYS_PKL)
        logger.info(f"所有流线已解析并缓存 ({len(resolved_pathways)}条): {self.config.RESOLVED_PATHWAYS_PKL}")
        return resolved_pathways
    
    def _resolve_single_template(self, template: Dict[str, Any]) -> List[Tuple[List[str], float]]:
        """
        根据单个流程模板，高效地解析出所有可能的具体流线及其权重。
        该方法能正确处理端点（如门）和核心路径节点权重的不同计算逻辑。

        Args:
            template (Dict[str, Any]): 单个就医流程的模板字典。

        Returns:
            List[Tuple[List[str], float]]: 一个包含(具体流线, 最终权重)元组的列表。
        """
        base_weight = template.get('base_weight', 1.0)

        # 1. 准备所有部分的选项列表
        start_nodes_options = [self.variants.get(gn, [gn]) for gn in template['start_nodes']]
        core_sequences_options = [self.variants.get(gn, [gn]) for gn in template['core_sequence']]
        end_nodes_options = [self.variants.get(gn, [gn]) for gn in template['end_nodes']]

        # 2. 一次性计算所有组合的笛卡尔积
        # 注意：我们将起点、核心和终点的组合分开处理，以便应用不同的权重逻辑
        start_combinations = list(itertools.product(*start_nodes_options))
        core_combinations = list(itertools.product(*core_sequences_options))
        end_combinations = list(itertools.product(*end_nodes_options))

        pathways = []
        # 3. 组合并计算权重 (恢复使用清晰的三重循环，因为逻辑已变得复杂)
        for start_combo in start_combinations:
            for core_combo in core_combinations:
                for end_combo in end_combinations:
                    
                    # --- 权重计算开始 ---
                    final_weight = base_weight
                    
                    # a. 计算起点的权重 (总是计算)
                    for node in start_combo:
                        final_weight *= self._get_normalized_weight(node)
                        
                    # b. 计算核心路径的权重 (每个通用名只计算一次)
                    processed_core_generics = set()
                    for node in core_combo:
                        generic_name = str(node).split('_')[0]
                        if generic_name not in processed_core_generics:
                            final_weight *= self._get_normalized_weight(node)
                            processed_core_generics.add(generic_name)

                    # c. 计算终点的权重 (总是计算)
                    for node in end_combo:
                        final_weight *= self._get_normalized_weight(node)

                    # --- 权重计算结束 ---

                    # 拼接成完整流线
                    full_path = list(start_combo) + list(core_combo) + list(end_combo)
                    pathways.append({
                        "process_id": template['process_id'],
                        "path": full_path,
                        "weight": final_weight
                    })
            
        return pathways
    
    def _get_normalized_weight(self, node_name: str) -> float:
        """
        计算单个节点的归一化流量权重。
        如果一个通用名下只有一个变体，其权重因子为1.0。
        """
        generic_name = str(node_name).split('_')[0]
        # 使用 self.traffic，它是在 __init__ 中加载或生成的
        distribution_map = self.traffic.get(generic_name)
        
        # 如果找不到分布或只有一个变体，则权重因子为1
        if not distribution_map or len(distribution_map) <= 1:
            return 1.0

        total_weight = sum(distribution_map.values())
        if total_weight == 0:
            logger.warning(f"通用名 '{generic_name}' 的总权重为0，无法进行归一化。")
            return 0.0

        raw_weight = distribution_map.get(node_name, 0.0)
        return raw_weight / total_weight