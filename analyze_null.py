import json
from src.config import RLConfig
from src.rl_optimizer.data.cache_manager import CacheManager

# 加载数据
with open('results/layouts/best_layouts_20250809-214240.json', 'r') as f:
    data = json.load(f)

# 获取科室列表
config = RLConfig()
cache = CacheManager(config)
all_departments = set(cache.placeable_departments)

print("=== 分析布局中的null问题 ===\n")

# 分析每个算法的布局
for algo_name in ['original', 'simulated_annealing', 'genetic_algorithm']:
    if algo_name not in data:
        continue
    
    if algo_name == 'original':
        layout = data[algo_name]['layout']
    else:
        layout = data[algo_name]['best_layout']
    
    print(f"{algo_name}:")
    print(f"  布局长度: {len(layout)}")
    
    # 统计null
    null_count = layout.count(None)
    null_indices = [i for i, x in enumerate(layout) if x is None]
    print(f"  null数量: {null_count}")
    if null_count > 0:
        print(f"  null位置: {null_indices}")
    
    # 检查科室完整性
    non_null_depts = [x for x in layout if x is not None]
    layout_depts = set(non_null_depts)
    
    # 找出缺失的科室
    missing_depts = all_departments - layout_depts
    if missing_depts:
        print(f"  缺失的科室: {missing_depts}")
    
    # 检查重复
    if len(non_null_depts) != len(layout_depts):
        from collections import Counter
        counter = Counter(non_null_depts)
        duplicates = [k for k, v in counter.items() if v > 1]
        print(f"  重复的科室: {duplicates}")
    
    print()

print("=== 结论 ===")
print(f"总共应有 {len(all_departments)} 个科室")
print(f"槽位数量: {len(cache.placeable_slots)}")
print(f"科室和槽位是否相同: {cache.placeable_departments == cache.placeable_slots}")