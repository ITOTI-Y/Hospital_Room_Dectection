#!/usr/bin/env python3
"""
测试修复后的算法是否还会产生null值
"""

import subprocess
import json
import sys
from pathlib import Path

def test_fixed_algorithms():
    """测试修复后的算法"""
    print("=== 测试修复后的算法 ===\n")
    
    # 运行算法对比
    cmd = [
        "uv", "run", "python", "main.py",
        "--mode", "compare",
        "--algorithms", "simulated_annealing,genetic_algorithm",
        "--max-iterations", "200",  # 使用较少迭代以加快测试
        "--population-size", "30"
    ]
    
    print(f"运行命令: {' '.join(cmd)}")
    print("这可能需要一些时间...\n")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"命令执行失败: {result.stderr}")
        return False
    
    print("命令执行成功，检查结果...\n")
    
    # 查找最新的布局文件
    layouts_dir = Path("./results/layouts")
    json_files = list(layouts_dir.glob("best_layouts_*.json"))
    if not json_files:
        print("未找到布局文件")
        return False
    
    latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
    print(f"检查文件: {latest_file}\n")
    
    # 加载并分析JSON
    with open(latest_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    has_null = False
    for algo_name in ['original', 'simulated_annealing', 'genetic_algorithm']:
        if algo_name not in data:
            continue
        
        if algo_name == 'original':
            layout = data[algo_name]['layout']
            cost = data[algo_name]['cost']
        else:
            layout = data[algo_name]['best_layout']
            cost = data[algo_name]['best_cost']
        
        null_count = layout.count(None)
        print(f"{algo_name}:")
        print(f"  布局长度: {len(layout)}")
        print(f"  成本: {cost:.2f}")
        print(f"  包含null: {'是' if null_count > 0 else '否'}")
        
        if null_count > 0:
            has_null = True
            null_indices = [i for i, x in enumerate(layout) if x is None]
            print(f"  null数量: {null_count}")
            print(f"  null位置: {null_indices}")
            
            # 找出缺失的科室
            all_depts = set(data['original']['layout'])
            layout_depts = set(x for x in layout if x is not None)
            missing = all_depts - layout_depts
            if missing:
                print(f"  缺失科室: {missing}")
        
        if algo_name != 'original' and 'improvement' in data[algo_name]:
            print(f"  改进率: {data[algo_name]['improvement']:.1f}%")
        print()
    
    if not has_null:
        print("✅ 测试通过：所有布局都不包含null值！")
        return True
    else:
        print("❌ 测试失败：仍有布局包含null值")
        return False

if __name__ == "__main__":
    success = test_fixed_algorithms()
    sys.exit(0 if success else 1)