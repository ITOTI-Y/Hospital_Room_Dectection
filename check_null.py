import json

# 检查模拟退火的结果
with open('results/layouts/best_layouts_20250809-214240.json', 'r') as f:
    data = json.load(f)

sa_layout = data['simulated_annealing']['best_layout']
print(f'模拟退火最优布局长度: {len(sa_layout)}')

# 找出null的位置
null_indices = [i for i, x in enumerate(sa_layout) if x is None]
print(f'null出现在索引: {null_indices}')

# 检查所有科室是否都在布局中
all_depts = [x for x in sa_layout if x is not None]
print(f'布局中非null科室数: {len(all_depts)}')

# 检查是否有重复
unique_depts = set(all_depts)
has_duplicates = len(all_depts) != len(unique_depts)
print(f'是否有重复科室: {has_duplicates}')

if has_duplicates:
    # 找出重复的科室
    from collections import Counter
    counter = Counter(all_depts)
    duplicates = [k for k, v in counter.items() if v > 1]
    print(f'重复的科室: {duplicates}')