# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 语言设置

**重要：Claude Code 在此项目中必须使用中文回复。**

## 项目概述

医院房间检测与布局优化系统 - 基于图像处理和多种优化算法的医院网络生成与布局优化工具。

### 核心架构

系统采用统一的双阶段架构，通过 `main.py` 统一入口管理所有功能：

1. **网络生成阶段**：
   - 从楼层平面图生成多层医院网络图
   - 图像处理：识别房间、走廊、门等区域
   - 网络构建：创建节点和边的图结构
   - 行程时间计算：生成房间间行程时间矩阵

2. **布局优化阶段**：
   - **PPO强化学习算法**：基于深度强化学习的布局优化
   - **模拟退火算法**：经典启发式优化算法
   - **遗传算法**：进化计算优化算法
   - 统一约束管理：面积匹配、固定位置、相邻性约束
   - 统一目标函数：CostCalculator成本计算器

3. **对比分析阶段**：
   - 多算法性能对比
   - 详细分析报告生成
   - 可视化图表输出

## 关键模块

- `src/core/`: 核心控制模块
  - `network_generator.py`: 网络生成器（整合SuperNetwork功能）
  - `algorithm_manager.py`: 算法管理器（统一管理所有优化算法）
- `src/algorithms/`: 优化算法模块
  - `base_optimizer.py`: 算法基类（统一接口）
  - `ppo_optimizer.py`: PPO强化学习优化器
  - `simulated_annealing.py`: 模拟退火优化器
  - `genetic_algorithm.py`: 遗传算法优化器
  - `constraint_manager.py`: 约束管理器（统一约束处理）
- `src/comparison/`: 对比分析模块
  - `results_comparator.py`: 结果对比分析器
- `src/network/`: 网络生成核心模块
  - `network.py`: 单层网络生成器
  - `super_network.py`: 多层网络管理器
  - `node_creators.py`: 各类节点创建策略
  - `node.py`: 节点定义类
  - `graph_manager.py`: 图管理器
- `src/rl_optimizer/`: 强化学习优化器
  - `env/cost_calculator.py`: 成本计算器（所有算法共用）
  - `env/layout_env.py`: 布局优化环境（支持动态基线奖励归一化）
  - `data/cache_manager.py`: 数据缓存管理
  - `utils/lr_scheduler.py`: 学习率调度器工具
  - `utils/shared_state_manager.py`: 跨进程共享状态管理器
  - `utils/reward_normalizer.py`: 动态基线奖励归一化器
  - `utils/baseline_monitor.py`: 基线变化监控和可视化工具
- `src/analysis/`: 分析工具
  - `travel_time.py`: 行程时间计算
  - `process_flow.py`: 就医流程分析
  - `word_detect.py`: 词汇检测工具
- `src/image_processing/`: 图像处理模块
- `src/plotting/`: 可视化模块
- `src/config.py`: 配置管理（NetworkConfig、RLConfig）

## 代码规范

### 配置管理
- 所有硬编码的配置项均应放置在config.py文件中

## 常用命令

### 环境管理
```bash
# 同步依赖环境
uv sync

# 检查项目状态
uv run python --version
```

### 统一命令行接口（通过main.py）

#### 1. 网络生成
```bash
# 生成多层医院网络和行程时间矩阵
uv run python main.py --mode network

# 自定义图像目录
uv run python main.py --mode network --image-dir ./data/label/

# 输出文件：
# - result/hospital_network_3d.html (网络可视化)
# - result/hospital_travel_times.csv (行程时间矩阵)
```

#### 2. 单算法优化
```bash
# 运行PPO强化学习算法
uv run python main.py --mode optimize --algorithm ppo

# 运行模拟退火算法
uv run python main.py --mode optimize --algorithm simulated_annealing --max-iterations 5000

# 运行遗传算法
uv run python main.py --mode optimize --algorithm genetic_algorithm --population-size 50

# 算法特定参数示例：
# PPO参数
uv run python main.py --mode optimize --algorithm ppo --total-timesteps 100000

# 从预训练模型继续训练
uv run python main.py --mode optimize --algorithm ppo \
  --pretrained-model-path /path/to/best_model.zip --total-timesteps 10000

# 模拟退火参数
uv run python main.py --mode optimize --algorithm simulated_annealing --initial-temperature 1500.0 --max-iterations 10000

# 遗传算法参数
uv run python main.py --mode optimize --algorithm genetic_algorithm --population-size 100 --max-iterations 500
```

#### 3. 多算法对比分析
```bash
# 运行所有算法进行对比
uv run python main.py --mode compare --algorithms ppo,simulated_annealing,genetic_algorithm

# 并行执行算法（谨慎使用，可能需要大量资源）
uv run python main.py --mode compare --algorithms simulated_annealing,genetic_algorithm --parallel

# 不生成图表和报告（仅获取基本对比结果）
uv run python main.py --mode compare --algorithms ppo,simulated_annealing --no-plots --no-report

# 详细输出
uv run python main.py --mode compare --algorithms ppo,simulated_annealing,genetic_algorithm --verbose
```

#### 4. 结果可视化
```bash
# 可视化算法结果（功能待实现）
uv run python main.py --mode visualize --results-file ./results/comparison/best_layouts_20231201-120000.json
```

### 输出文件结构
```
result/
├── hospital_network_3d.html           # 网络可视化
├── hospital_travel_times.csv          # 行程时间矩阵
├── comparison/                         # 算法对比结果
│   ├── algorithm_comparison_*.csv     # 对比表格
│   ├── *_result_*.json               # 详细算法结果
│   └── best_layouts_*.json           # 最优布局对比
├── plots/                             # 对比图表
│   ├── cost_comparison_*.png         # 成本对比图
│   ├── time_comparison_*.png         # 时间对比图
│   ├── convergence_curves_*.png      # 收敛曲线图
│   ├── performance_radar_*.png       # 性能雷达图
│   └── algorithm_heatmap_*.png       # 算法特性热力图
└── reports/                           # 分析报告
    └── algorithm_comparison_report_*.md

logs/
├── hospital_optimizer.log            # 系统运行日志
└── ppo_layout_*/                     # PPO训练日志（如果使用PPO算法）
    ├── final_model.zip
    ├── checkpoints/
    └── tensorboard_logs/
```

## 数据流程

### 输入数据
- `data/image/`: 楼层平面原图 (1F.png, 2F.png等)
- `data/label/`: 楼层标注图 (-1F-meng.png, 1F-meng.png等)
- `src/rl_optimizer/data/process_templates.json`: 就医流程模板

### 处理流程
1. 图像预处理 → 颜色映射识别 → 区域分割
2. 节点创建 → 边连接 → 网络图构建
3. 行程时间计算 → 缓存生成
4. 强化学习训练 → 布局优化

### 输出结果
- `result/`: 网络可视化和分析结果
- `logs/`: 训练日志和优化模型
- `src/rl_optimizer/data/cache/`: 中间缓存文件

## 技术栈

- **图像处理**: OpenCV, PIL
- **网络分析**: NetworkX
- **机器学习**: PyTorch, Stable-baselines3 (MaskablePPO)
- **可视化**: Plotly, Matplotlib
- **数据处理**: Pandas, NumPy
- **依赖管理**: uv (Python包管理器)
- **其他**: scikit-learn, sentence-transformers

## 配置文件

### NetworkConfig (src/config.py:64-135)
- 网络生成参数
- 图像处理设置
- 可视化配置

### RLConfig (src/config.py:136-244)
- 强化学习超参数
- 训练配置
- 约束设置
- **学习率调度器配置**: 支持线性衰减和常数学习率
- **断点续训配置**: 支持训练中断恢复和checkpoint管理

## 重要开发注意事项

### 颜色映射系统
- 项目使用RGB颜色映射来识别不同的医院区域类型
- COLOR_MAP定义在`src/config.py:7-62`，包含62种不同的区域类型
- 每个颜色对应特定的医院功能区域（如妇科、急诊科、走廊等）
- 修改颜色映射前需确保与标注图像一致

### 数据预处理要求
- 楼层图像需要180度旋转（IMAGE_ROTATE=180）
- 最小区域阈值为60像素（AREA_THRESHOLD=60）
- 网格大小为40像素（GRID_SIZE=40）

### 性能考虑
- 多进程支持：SuperNetwork支持并行处理多个楼层
- 缓存机制：RL优化器使用缓存来加速重复计算
- GPU支持：PyTorch模型支持CUDA加速
- **学习率调度**: 支持线性衰减学习率提高训练稳定性和收敛性
- **断点续训**: 支持长时间训练的中断恢复，定期保存checkpoint防止训练丢失

### 断点续训机制
- **自动checkpoint**: 训练过程中自动保存模型状态和训练元数据
- **智能恢复**: 可自动查找最新checkpoint或指定特定checkpoint恢复
- **状态完整性**: 保存模型、优化器、学习率调度器等完整训练状态
- **进度跟踪**: 准确计算剩余训练步数，支持精确的训练进度恢复

## 动态文档更新机制

**重要：此文档应在每次重要操作后更新，以保持准确性和实用性。**

### 更新触发条件
- 添加新功能或模块
- 修改核心配置或流程
- 发现新的使用模式或最佳实践
- 解决常见问题
- 性能优化或重构

### 更新指导原则
1. **保持简洁性**: 只记录需要跨文件理解的架构信息
2. **命令准确性**: 确保所有命令都使用正确的uv格式
3. **中文交互**: 所有说明和注释使用中文
4. **实用导向**: 专注于实际开发中需要的信息
5. **版本感知**: 记录重要的版本变化和兼容性信息

### 如何更新
在完成以下操作后，请更新此文档的相应部分：
- 修改命令或工作流程 → 更新"常用命令"部分
- 新增模块或重构 → 更新"关键模块"和"项目概述"
- 配置变更 → 更新"配置文件"部分
- 发现问题和解决方案 → 在相关部分添加注释

## 最近更新记录

- 2025-09-04: **实现动态基线奖励归一化系统** - 解决PPO训练中奖励尺度不一致问题，实现智能的相对改进奖励机制：
  - 新增SharedStateManager跨进程共享状态管理器，支持多并行环境的EMA基线维护
  - 创建RewardNormalizer奖励归一化器，实现各奖励组件的标准化和相对改进计算
  - 修改LayoutEnv环境，集成动态基线奖励计算，支持传统模式和归一化模式双模切换
  - 更新PPO优化器，完整集成共享状态管理和归一化奖励统计
  - 新增BaselineMonitor基线监控器，提供实时监控、历史追踪和可视化分析功能
  - 新增配置参数：ENABLE_DYNAMIC_BASELINE、EMA_ALPHA、BASELINE_WARMUP_EPISODES等
  - 核心优势：自适应奖励缩放、相对改进激励、训练稳定性提升、跨环境状态同步
  - 完整测试验证：通过test_dynamic_baseline.py全面验证系统功能正确性
- 2025-08-05: 初始创建动态CLAUDE.md文档，包含完整架构概述和uv命令格式
- 2025-08-05: 添加重要开发注意事项，包含颜色映射系统、数据预处理要求和性能考虑
- 2025-08-05: 完善技术栈信息，添加MaskablePPO等具体实现细节
- 2025-08-05: **引入线性衰减学习率调度器** - 添加lr_scheduler.py工具模块，支持线性衰减和常数学习率调度，提高训练稳定性和收敛性
- 2025-08-05: 添加配置管理规范：所有硬编码的配置项均应放置在config.py文件中
- 2025-08-05: **实现完整断点续训功能** - 添加CheckpointCallback类、扩展RLConfig配置、增强PPOAgent模型恢复能力、更新命令行接口，支持训练中断恢复和精确进度跟踪
- 2025-08-05: **重大架构重构** - 统一所有算法运行接口到main.py，实现多算法对比研究框架：
  - 删除rl_main.py，整合所有功能到统一入口
  - 新增传统启发式算法：模拟退火(SimulatedAnnealing)和遗传算法(GeneticAlgorithm)
  - 创建统一优化器基类BaseOptimizer，确保所有算法使用相同的CostCalculator目标函数
  - 实现ConstraintManager统一约束管理（面积、固定位置、相邻性约束）
  - 新增AlgorithmManager算法管理器，支持串行/并行多算法执行
  - 实现ResultsComparator结果对比分析器，生成详细对比表格、图表和报告
  - 重构NetworkGenerator类整合网络生成功能
  - 建立完整的目录结构：src/core/, src/algorithms/, src/comparison/
  - 实现统一命令行接口，支持network/optimize/compare/visualize四种运行模式
  - 添加全面的算法性能对比功能：成本、时间、收敛性、稳定性等多维度分析
- 2025-08-05: **修复启发式算法配置项无效问题** - 解决模拟退火和遗传算法命令行参数无法生效的关键问题：
  - 修改AlgorithmManager._create_optimizer()方法，使其正确接受并传递构造函数参数
  - 更新run_single_algorithm()方法，分离构造参数和运行参数的传递逻辑
  - 修复参数传递机制：构造函数参数在算法实例化时传递，运行时参数在optimize()调用时传递
  - 验证修复效果：模拟退火算法的initial_temperature、max_iterations等参数现在能正确生效
  - 验证修复效果：遗传算法的population_size、max_iterations等参数现在能正确生效
  - 确保所有启发式算法的命令行参数配置现在完全可用
- 2025-08-05: **修复多进程环境日志重复问题** - 彻底解决Stable-baselines3多进程训练时的日志重复显示：
  - 修复setup_logger函数：添加严格的handler重复检查、设置propagate=False防止日志传播
  - 实现进程区分机制：只有主进程输出详细INFO级别日志，子进程仅输出WARNING以上级别
  - 保持所有监控功能完整性：TensorBoard记录等功能正常工作
  - 验证修复效果：多进程测试确认无重复日志输出，训练过程日志清晰可读
- 2025-08-05: **确保显示原始时间成本值** - 修正训练日志显示缩放reward值而非原始时间成本的问题：
  - 修改LayoutEnv._get_info()：明确传递原始时间成本和缩放reward，添加调试日志验证数据流
  - 恢复训练进度条显示：支持总episodes目标设置，实时显示训练进度百分比和可视化进度条
  - 改进日志内容：时间成本以"秒"为单位显示，同时提供对应的缩放reward用于对比验证
  - 增强最佳结果记录：发现新最佳布局时同时显示原始时间成本和对应缩放reward
  - 验证功能正确性：通过完整测试确认显示的是CostCalculator.calculate_total_cost()的原始值
- 2025-08-05: **修复算法对比雷达图生成错误** - 解决ResultsComparator中"IndexError: single positional indexer is out-of-bounds"的关键bug：
  - 修复_plot_performance_radar()函数中算法名称查找逻辑不匹配问题
  - self.algorithm_names使用results的keys而DataFrame使用result.algorithm_name导致查找失败
  - 改为直接使用DataFrame中实际算法名称进行数据查找和图表生成
  - 添加完整的防御性检查：空数据检查、NaN值处理、异常捕获等
  - 优化matplotlib中文字体支持，减少字体警告但不影响功能
  - 验证修复效果：算法对比功能完全正常，所有图表包括雷达图成功生成
  - 创建EpisodeInfoVecEnvWrapper：自定义VecEnv包装器确保episode信息在多进程环境中正确传递
  - 修复PPOOptimizer环境创建：在训练和评估环境中都使用episode信息包装器
  - 添加详细调试日志：在环境、包装器、回调等关键位置添加DEBUG级别日志便于问题诊断
  - 确保数据完整性：包装器自动补充标准episode字段（r、l）并保持自定义字段（time_cost、layout等）
  - 解决核心问题：现在训练过程能正确显示原始时间成本而不是仅显示抽象reward值
- 2025-08-09: **彻底优化多进程训练日志输出** - 删除RL模型子进程的冗余logger显示，显著提升训练日志可读性：
  - 恢复LayoutEnv正常日志级别：移除临时DEBUG级别配置，改回默认INFO级别，减少环境层面的详细调试输出
  - 强化VecEnvWrapper进程隔离：添加multiprocessing进程检查，确保只有主进程输出详细调试日志
  - 添加严格的日志级别检查：所有DEBUG级别日志都增加logger.isEnabledFor(10)检查，确保只在需要时输出
  - 验证修复效果：多进程测试确认子进程只输出WARNING以上级别日志，主进程保持完整的训练监控功能
  - 保持功能完整性：TensorBoard记录等核心功能完全正常
  - 显著改善用户体验：PPO训练过程日志现在清晰易读，无重复输出，专注于关键训练进度和性能指标
- 2025-08-09: **删除训练指标回调类** - 简化PPO训练流程，移除不再需要的训练指标监控组件：
  - 删除 src/rl_optimizer/utils/training_metrics_callback.py 文件（351行代码）
  - 修改PPOOptimizer：移除TrainingMetricsCallback的导入、创建和使用
  - 清理相关引用：删除metrics_callback相关的日志输出和统计信息显示
  - 保持核心功能：PPO算法训练功能完全保留，仅移除额外的指标监控层
  - 简化日志输出：训练过程专注于算法本身的基础信息，减少冗余显示
- 2025-08-09: **代码清理和文档重构** - 完成项目结构优化和文档重写：
  - 删除无用文件和目录：debug/、layout_opt.prof、data/model/等
  - 移除废弃模块：src/graph/和src/optimization/
  - 创建简化的图管理组件：src/network/node.py和graph_manager.py
  - 修复所有导入引用，更新为新的模块路径
  - 重写README.md，基于实际代码结构提供准确的使用指南
  - 保留所有有用功能模块，包括词汇检测等分析工具
  - 验证系统功能正常，确保清理后所有命令正常工作
- 2025-08-09: **引入势函数面积匹配奖励机制** - 在强化学习布局优化中添加面积匹配软奖励：
  - 新增配置参数：在RLConfig中添加`AREA_MATCH_REWARD_WEIGHT`（0.2）和`AREA_MATCH_BONUS_BASE`（10.0）参数
  - 实现面积匹配度计算：新增`_calculate_area_match_score()`方法，计算科室与槽位的面积匹配度（0-1之间）
  - 增强势函数：`_calculate_potential()`现在包含两部分：时间成本势函数 + 面积匹配奖励
  - 面积匹配度算法：使用相对差异（而非绝对差异）评估匹配度，更公平地处理不同规模房间
  - 添加统计监控：在episode结束时报告平均、最小、最大面积匹配度
  - 验证功能：通过测试脚本确认面积匹配奖励正确集成到势函数中，不直接影响step奖励
  - 预期效果：智能体将在满足硬约束基础上，优先选择面积更匹配的科室-槽位配对，提升空间利用效率
- 2025-08-10: **实现PPO最佳模型保存和评估功能** - 完善训练过程中的模型保存和评估机制：
  - 修复_evaluate_best_model()方法：从简单返回随机布局改为真正加载和评估最佳模型
  - 添加模型加载降级方案：优先使用best_model，找不到时降级到final_model或最新checkpoint
  - 实现多episode评估：评估5个episode并选择最优结果，提高评估稳定性
  - 修复动作掩码问题：确保推理时正确传递action_masks参数，避免选择已放置科室
  - 创建完整推理工具：新增inference_ppo_model.py，支持单次/多次推理和详细结果保存
- 2025-08-10: **增强推理工具面积匹配显示** - 在PPO模型推理中添加面积匹配度计算和显示：
  - 推理过程计算面积匹配度：为每个科室-槽位配对计算0-1之间的匹配分数
  - 统计信息展示：显示平均、最小、最大面积匹配度
  - 详细不匹配报告：记录匹配度低于0.8的分配情况
  - 结果文件包含面积信息：JSON和文本输出都包含完整的面积匹配统计
  - 验证效果：模型实现了99.1%的平均面积匹配度，证明面积奖励机制有效
- 2025-08-10: **支持从预训练模型继续训练** - 实现PPO算法从已有最佳模型继续训练的功能：
  - 修改PPOOptimizer构造函数：添加pretrained_model_path可选参数
  - 更新模型加载逻辑：支持从指定的预训练模型文件加载权重继续训练
  - 扩展命令行接口：添加--pretrained-model-path参数支持指定模型路径
  - 修改AlgorithmManager：正确传递预训练模型路径到PPO优化器
  - 使用示例：`uv run python main.py --mode optimize --algorithm ppo --pretrained-model-path /path/to/best_model.zip --total-timesteps 10000`
  - 验证功能：成功从已训练模型继续训练，保持原有性能水平
- 2025-08-11: **改进继续训练的最佳模型保存机制** - 确保只有性能真正提升时才保存新的最佳模型：
  - 创建PretrainedEvalCallback类：扩展EvalCallback，支持预评估预训练模型性能
  - 预训练模型评估：在_init_callback中加载并评估预训练模型，获取其平均reward
  - 设置初始基准：将预训练模型的性能作为best_mean_reward的初始值（而非-inf）
  - 智能回调选择：PPOOptimizer根据是否有预训练模型自动选择合适的评估回调
  - 关键优势：避免性能下降的模型被错误标记为"最佳"，确保模型质量持续提升
  - 向后兼容：正常训练（无预训练模型）时行为完全不变
  - 验证效果：测试确认预训练模型性能（-99.00）被正确设置为初始基准
- 2025-08-12: **彻底重构遗传算法实现约束感知优化** - 基于产品经理架构师解决方案，完成遗传算法约束问题的全面修复：
  - 修复constraint_manager.py中的重复方法定义bug，解决代码冲突问题
  - 实现SmartConstraintRepairer智能约束修复器：支持贪心面积匹配、交换优化、随机修复三种策略
  - 重构遗传算法为约束感知版本：集成FPX(Fixed Position Crossover)约束感知交叉算子
  - 实现约束导向的智能变异策略：根据约束违反情况进行有针对性的修复性变异
  - 增强初始种群生成：30%贪心面积匹配 + 20%种子变异 + 50%约束感知随机个体
  - 实现自适应参数调整机制：根据停滞情况和种群多样性动态调整变异率和交叉率
  - 增强种群多样性维护策略：多层次多样性检测和个体替换机制
  - 添加全面的性能统计和监控：约束违反率、修复成功率、算法特定指标等
  - 更新config.py配置参数：优化遗传算法默认参数，添加约束感知相关配置
  - 验证测试通过：结构完整性测试、功能测试、集成测试，改进率达到39.1%
  - 关键技术突破：FPX交叉算子保持固定位置约束、匈牙利算法最优分配、多策略约束修复
- 2025-08-15: **实施PPO算法相邻性奖励性能优化方案** - 完成相邻性奖励计算的全面性能优化，显著提升训练速度：
  - 创建OptimizedAdjacencyRewardCalculator优化计算器：实现向量化计算、稀疏矩阵优化、多级缓存机制
  - 核心优化策略：向量化NumPy广播替代嵌套循环、COO格式稀疏矩阵减少内存开销、LRU缓存+预计算缓存
  - 集成LayoutEnv环境：支持优化计算器和传统方法的无缝切换，保持完全API兼容性
  - 添加配置开关：ENABLE_ADJACENCY_OPTIMIZATION控制新旧版本切换，支持渐进式部署
  - 扩展配置参数：添加稀疏矩阵阈值、向量化批处理大小、内存优化等精细化配置选项
  - 实现性能监控：AdjacencyMetrics数据类提供详细的执行时间、缓存命中率、内存使用统计
  - 创建基准测试脚本：benchmark_adjacency_performance.py全面对比新旧方法的性能表现
  - 添加验证脚本：test_adjacency_optimization.py确保优化功能正确性和结果一致性
  - 技术突破：矩阵索引优化减少查找开销、三种相邻性计算合并为单次遍历、批量处理提升效率
  - 预期效果：相邻性奖励计算速度提升2-5倍，内存使用减少30-50%，训练过程更加流畅