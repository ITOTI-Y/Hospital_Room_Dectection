#!/usr/bin/env python3
"""
加载已训练的PPO最佳模型并进行推理
"""

import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from src.rl_optimizer.env.layout_env import LayoutEnv
from src.rl_optimizer.env.cost_calculator import CostCalculator
from src.rl_optimizer.data.cache_manager import CacheManager
from src.algorithms.constraint_manager import ConstraintManager
from src.config.config import RLConfig
from src.rl_optimizer.utils.setup import setup_logger, save_json

logger = setup_logger(__name__)


class PPOModelInference:
    """PPO模型推理器"""

    def __init__(self, model_path: str, config: Optional[RLConfig] = None):
        """
        初始化推理器

        Args:
            model_path: 模型文件路径
            config: 配置对象（可选）
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        self.config = config or RLConfig()
        logger.info("正在初始化PPO模型推理器...")
        logger.info(f"模型路径: {self.model_path}")

        # 初始化必要的组件
        self._setup_components()

        # 加载模型
        self._load_model()

    def _setup_components(self):
        """设置必要的组件"""
        # 初始化缓存管理器
        self.cache_manager = CacheManager(self.config)

        # 加载行程时间数据
        travel_times_path = self.config.TRAVEL_TIMES_CSV
        if travel_times_path.exists():
            self.travel_times = pd.read_csv(travel_times_path, index_col=0)
            logger.info(f"加载行程时间矩阵: {travel_times_path}")
        else:
            raise FileNotFoundError(f"行程时间文件不存在: {travel_times_path}")

        # 初始化成本计算器
        self.cost_calculator = CostCalculator(
            config=self.config,
            resolved_pathways=self.cache_manager.resolved_pathways,
            travel_times=self.travel_times,
            placeable_slots=self.cache_manager.placeable_slots,
            placeable_departments=self.cache_manager.placeable_departments,
        )

        self.origin_total_cost = self.cost_calculator.calculate_total_cost(
            self.travel_times.columns.to_list()
        )

        # 初始化约束管理器
        self.constraint_manager = ConstraintManager(
            config=self.config, cache_manager=self.cache_manager
        )

        # 环境参数
        self.env_kwargs = {
            "config": self.config,
            "cache_manager": self.cache_manager,
            "cost_calculator": self.cost_calculator,
            "constraint_manager": self.constraint_manager,
        }

    def _load_model(self):
        """加载PPO模型"""
        logger.info("正在加载模型...")

        # 加载模型（不需要环境，仅用于推理）
        self.model = MaskablePPO.load(
            str(self.model_path), device="cuda" if torch.cuda.is_available() else "cpu"
        )

        device = "CUDA" if torch.cuda.is_available() else "CPU"
        logger.info(f"✓ 模型加载成功，使用设备: {device}")

    def infer_single_episode(self, render: bool = False) -> Dict[str, Any]:
        """
        执行单次推理

        Args:
            render: 是否显示推理过程

        Returns:
            推理结果字典
        """
        # 创建环境（使用ActionMasker包装）
        env = ActionMasker(LayoutEnv(**self.env_kwargs), LayoutEnv._action_mask_fn)

        # 重置环境
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        terminated = False
        step_count = 0
        actions_taken = []

        logger.info("开始推理...")

        while not terminated:
            # 获取当前的动作掩码
            action_mask = (
                env.env.get_action_mask()
                if hasattr(env, "env")
                else env.get_action_mask()
            )

            # 使用模型预测下一步动作（传递action_masks）
            action, _ = self.model.predict(
                obs, action_masks=action_mask, deterministic=True
            )
            actions_taken.append(int(action))

            # 执行动作
            result = env.step(int(action))
            if len(result) == 5:
                obs, reward, terminated, _, info = result
            else:
                obs, reward, terminated, info = result[:4]

            step_count += 1

            if render and step_count % 10 == 0:
                logger.info(f"  步骤 {step_count}: 动作={action}, 奖励={reward:.2f}")

        # 获取最终布局
        inner_env = env.env if hasattr(env, "env") else env
        final_layout = inner_env._get_final_layout_str()

        # 检查是否存在空余槽位
        if None in final_layout:
            empty_slots = final_layout.count(None)
            logger.warning(f"推理结果存在 {empty_slots} 个空余槽位，不符合完整布局要求")
            return None  # 返回None表示推理失败

        # 计算成本
        total_cost = self.cost_calculator.calculate_total_cost(final_layout)
        per_process_cost = self.cost_calculator.calculate_per_process_cost(final_layout)

        # 构建布局映射和计算面积匹配度
        layout_map = {}
        area_match_scores = []
        area_mismatches = []

        # 获取面积映射
        dept_areas_map = inner_env.dept_areas_map  # 科室ID到面积的映射
        slot_areas = inner_env.slot_areas  # 槽位面积数组

        for i, (slot, dept) in enumerate(zip(inner_env.placeable_slots, final_layout)):
            if dept is not None:
                layout_map[slot] = dept

                # 计算面积匹配度
                slot_area = slot_areas[i]  # 使用索引获取槽位面积
                dept_area = dept_areas_map.get(dept, slot_area)  # 获取科室面积

                # 使用相对差异计算匹配度（0-1之间，1表示完美匹配）
                area_diff_ratio = abs(dept_area - slot_area) / max(dept_area, slot_area)
                match_score = max(0, 1 - area_diff_ratio)
                area_match_scores.append(match_score)

                # 记录不匹配的情况（匹配度低于0.8）
                if match_score < 0.8:
                    area_mismatches.append(
                        {
                            "department": dept,
                            "slot": slot,
                            "dept_area": dept_area,
                            "slot_area": slot_area,
                            "match_score": match_score,
                        }
                    )

        # 计算面积匹配统计
        avg_area_match = np.mean(area_match_scores) if area_match_scores else 0
        min_area_match = min(area_match_scores) if area_match_scores else 0
        max_area_match = max(area_match_scores) if area_match_scores else 0

        result = {
            "layout": final_layout,
            "layout_map": layout_map,
            "total_cost": total_cost,
            "improve_ratio": total_cost / self.origin_total_cost
            if self.origin_total_cost
            else 0,
            "per_process_cost": per_process_cost,
            "steps": step_count,
            "actions": actions_taken,
            "final_reward": reward,
            "area_match_stats": {
                "average": avg_area_match,
                "min": min_area_match,
                "max": max_area_match,
                "mismatches": area_mismatches,
            },
        }

        logger.info(f"✓ 推理完成！总步数: {step_count}, 总成本: {total_cost:.2f}")
        logger.info(
            f"  面积匹配度 - 平均: {avg_area_match:.3f}, 最小: {min_area_match:.3f}, 最大: {max_area_match:.3f}"
        )

        # 在verbose模式下显示面积不匹配的详细信息
        if render and area_mismatches:
            logger.info("\n  面积不匹配的分配（匹配度<0.8）:")
            for mismatch in area_mismatches[:5]:  # 只显示前5个
                logger.info(
                    f"    {mismatch['department']:20s} -> {mismatch['slot']:20s}: "
                    f"科室面积={mismatch['dept_area']:.0f}, 槽位面积={mismatch['slot_area']:.0f}, "
                    f"匹配度={mismatch['match_score']:.3f}"
                )
            if len(area_mismatches) > 5:
                logger.info(f"    ... 还有 {len(area_mismatches) - 5} 个不匹配的分配")

        return result

    def infer_multiple_episodes(self, n_episodes: int = 5) -> Dict[str, Any]:
        """
        执行多次推理并返回最佳结果

        Args:
            n_episodes: 推理次数

        Returns:
            最佳结果和所有结果的统计信息
        """
        logger.info(f"执行 {n_episodes} 次推理...")

        results = []
        best_result = None
        best_cost = float("inf")
        failed_episodes = 0

        for i in range(n_episodes):
            logger.info(f"\n--- Episode {i + 1}/{n_episodes} ---")
            result = self.infer_single_episode(render=False)

            # 跳过无效结果（存在空余槽位的推理）
            if result is None:
                failed_episodes += 1
                logger.warning(f"  Episode {i + 1} 推理失败（存在空余槽位）")
                continue

            results.append(result)

            if result["total_cost"] < best_cost:
                best_cost = result["total_cost"]
                best_result = result
                logger.info(
                    f"  ★ 发现更优布局！成本: {best_cost:.2f}, 面积匹配度: {result['area_match_stats']['average']:.3f}"
                )

        # 检查是否有有效结果
        if not results:
            logger.error(f"所有 {n_episodes} 次推理均失败，无法生成完整布局")
            return {
                "best_result": None,
                "all_results": [],
                "statistics": {
                    "n_episodes": n_episodes,
                    "failed_episodes": failed_episodes,
                    "valid_episodes": 0,
                    "success_rate": 0.0,
                },
            }

        # 计算统计信息
        costs = [r["total_cost"] for r in results]
        area_matches = [r["area_match_stats"]["average"] for r in results]
        valid_episodes = len(results)
        success_rate = valid_episodes / n_episodes

        stats = {
            "best_result": best_result,
            "all_results": results,
            "statistics": {
                "min_cost": min(costs),
                "max_cost": max(costs),
                "mean_cost": np.mean(costs),
                "std_cost": np.std(costs),
                "n_episodes": n_episodes,
                "valid_episodes": valid_episodes,
                "failed_episodes": failed_episodes,
                "success_rate": success_rate,
                "mean_area_match": np.mean(area_matches),
                "std_area_match": np.std(area_matches),
            },
        }

        logger.info("\n" + "=" * 60)
        logger.info("推理统计:")
        logger.info(f"  总推理次数: {n_episodes}")
        logger.info(f"  有效推理次数: {valid_episodes}")
        logger.info(f"  失败推理次数: {failed_episodes}")
        logger.info(f"  成功率: {success_rate:.1%}")
        logger.info(f"  最优优化比例: {stats['best_result']['improve_ratio']:.1%}")
        logger.info(f"  最小成本: {stats['statistics']['min_cost']:.2f}")
        logger.info(f"  最大成本: {stats['statistics']['max_cost']:.2f}")
        logger.info(f"  平均成本: {stats['statistics']['mean_cost']:.2f}")
        logger.info(f"  标准差: {stats['statistics']['std_cost']:.2f}")
        logger.info(
            f"  平均面积匹配度: {stats['statistics']['mean_area_match']:.3f} (±{stats['statistics']['std_area_match']:.3f})"
        )
        logger.info("=" * 60)

        return stats

    def save_results(self, results: Dict[str, Any], output_path: str = None):
        """
        保存推理结果

        Args:
            results: 推理结果
            output_path: 输出路径
        """
        # 检查是否存在有效结果
        if results.get("best_result") is None:
            logger.warning("没有有效的推理结果，跳过保存")
            return

        if output_path is None:
            output_path = (
                Path("results")
                / "inference"
                / f"ppo_inference_{Path(self.model_path).parent.parent.name}.json"
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 转换numpy类型为Python类型
        def convert_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            return obj

        results = convert_types(results)
        save_json(results, str(output_path))
        logger.info(f"结果已保存到: {output_path}")

        # 同时保存布局映射为易读格式
        if "best_result" in results:
            layout_path = output_path.parent / f"layout_{output_path.stem}.txt"
            with open(layout_path, "w", encoding="utf-8") as f:
                f.write("最优布局方案\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"总成本: {results['best_result']['total_cost']:.2f}\n")

                # 添加面积匹配统计
                if "area_match_stats" in results["best_result"]:
                    stats = results["best_result"]["area_match_stats"]
                    f.write("\n面积匹配度统计:\n")
                    f.write(f"  平均匹配度: {stats['average']:.3f}\n")
                    f.write(f"  最小匹配度: {stats['min']:.3f}\n")
                    f.write(f"  最大匹配度: {stats['max']:.3f}\n")
                    f.write(f"  不匹配数量: {len(stats['mismatches'])}\n")

                f.write("\n槽位分配:\n")
                f.write("-" * 40 + "\n")
                for slot, dept in results["best_result"]["layout_map"].items():
                    f.write(f"{slot:30s} -> {dept}\n")

                # 添加面积不匹配的详细信息
                if "area_match_stats" in results["best_result"] and stats["mismatches"]:
                    f.write("\n\n面积不匹配的分配（匹配度<0.8）:\n")
                    f.write("-" * 60 + "\n")
                    for mismatch in stats["mismatches"]:
                        f.write(
                            f"{mismatch['department']:25s} -> {mismatch['slot']:25s}: "
                            f"科室={mismatch['dept_area']:6.0f}, 槽位={mismatch['slot_area']:6.0f}, "
                            f"匹配度={mismatch['match_score']:.3f}\n"
                        )
            logger.info(f"布局方案已保存到: {layout_path}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="PPO模型推理工具")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/pan/code/Hospital_Room_Dectection/results/model/ppo_layout_20250822-215147/best_model/best_model.zip",
        help="模型文件路径",
    )
    parser.add_argument("--n-episodes", type=int, default=1, help="推理次数（默认1次）")
    parser.add_argument("--single", action="store_true", help="仅执行单次推理")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径")
    parser.add_argument("--verbose", action="store_true", help="显示详细信息")

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("PPO模型推理工具")
    logger.info("=" * 80)

    try:
        # 创建推理器
        inferencer = PPOModelInference(args.model_path)

        # 执行推理
        if args.single:
            logger.info("\n执行单次推理...")
            result = inferencer.infer_single_episode(render=args.verbose)

            # 处理推理失败的情况
            if result is None:
                logger.error("单次推理失败（存在空余槽位），无法生成完整布局")
                results = {
                    "best_result": None,
                    "statistics": {
                        "n_episodes": 1,
                        "valid_episodes": 0,
                        "failed_episodes": 1,
                        "success_rate": 0.0,
                    },
                }
            else:
                # 包装为统一格式
                results = {
                    "best_result": result,
                    "statistics": {
                        "n_episodes": 1,
                        "valid_episodes": 1,
                        "failed_episodes": 0,
                        "success_rate": 1.0,
                        "min_cost": result["total_cost"],
                        "max_cost": result["total_cost"],
                        "mean_cost": result["total_cost"],
                        "std_cost": 0.0,
                        "mean_area_match": result["area_match_stats"]["average"],
                        "std_area_match": 0.0,
                    },
                }
        else:
            results = inferencer.infer_multiple_episodes(n_episodes=args.n_episodes)

        # 保存结果
        inferencer.save_results(results, args.output)

        logger.info("\n" + "=" * 80)
        logger.info("✅ 推理完成！")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"推理过程中发生错误: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
