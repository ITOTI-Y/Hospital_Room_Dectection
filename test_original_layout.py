#!/usr/bin/env python3
"""
测试原始布局功能
"""

import json
import subprocess
import sys
from pathlib import Path

def run_test():
    """运行测试"""
    print("测试原始布局功能...")
    
    # 运行模拟退火算法作为测试
    cmd = [
        "uv", "run", "python", "main.py",
        "--mode", "optimize",
        "--algorithm", "simulated_annealing",
        "--max-iterations", "100"  # 使用较少迭代以加快测试
    ]
    
    print(f"运行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"命令执行失败: {result.stderr}")
        return False
    
    print("命令执行成功")
    
    # 查找生成的JSON文件
    results_dir = Path("./results/comparison")
    if not results_dir.exists():
        print(f"结果目录不存在: {results_dir}")
        return False
    
    # 获取最新的结果文件
    json_files = list(results_dir.glob("simulated_annealing_result_*.json"))
    if not json_files:
        print("未找到结果文件")
        return False
    
    latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
    print(f"检查结果文件: {latest_file}")
    
    # 加载并检查JSON内容
    with open(latest_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 检查是否包含原始布局信息
    has_original_layout = 'original_layout' in data
    has_original_cost = 'original_cost' in data
    
    print(f"\n检查结果:")
    print(f"  包含original_layout: {has_original_layout}")
    print(f"  包含original_cost: {has_original_cost}")
    
    if has_original_layout and has_original_cost:
        print(f"  原始成本: {data['original_cost']:.2f}")
        print(f"  最优成本: {data['best_cost']:.2f}")
        
        if data['original_cost'] > 0:
            improvement = ((data['original_cost'] - data['best_cost']) / data['original_cost']) * 100
            print(f"  改进率: {improvement:.1f}%")
        
        print("\n✅ 测试通过：原始布局功能正常工作")
        return True
    else:
        print("\n❌ 测试失败：结果中未包含原始布局信息")
        return False

if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)