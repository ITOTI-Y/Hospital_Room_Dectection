# 医院房间检测与布局优化系统

基于图像处理和多种优化算法的医院网络生成与布局优化工具，提供从楼层平面图到最优布局方案的完整解决方案。

## 📋 项目概述

本系统采用双阶段架构，实现医院楼层平面图的自动分析和布局优化：

1. **网络生成阶段**：从楼层平面图自动识别房间、走廊、门等区域，构建多层医院网络图并计算行程时间矩阵
2. **布局优化阶段**：使用PPO强化学习、模拟退火、遗传算法等多种优化方法寻找最优布局方案

## ✨ 核心功能

### 🔍 智能网络生成
- **图像识别**：自动识别62种医院区域类型（妇科、急诊科、走廊、门等）
- **网络构建**：基于NetworkX生成多层医院网络图
- **行程时间计算**：生成房间间的准确行程时间矩阵
- **3D可视化**：生成交互式3D网络可视化图

### 🤖 多算法优化
- **PPO强化学习**：基于MaskablePPO的深度强化学习优化
- **模拟退火算法**：经典启发式优化方法
- **遗传算法**：进化计算优化方法
- **统一约束管理**：面积匹配、固定位置、相邻性等约束处理

### 📊 性能对比分析
- **多算法对比**：同时运行多个算法进行性能比较
- **详细分析报告**：自动生成Markdown格式的分析报告
- **可视化图表**：成本对比、收敛曲线、性能雷达图等

## 🏗️ 系统架构

```
src/
├── main.py              # 统一入口程序
├── config.py            # 系统配置管理
├── core/                # 核心控制模块
│   ├── network_generator.py    # 网络生成控制器
│   └── algorithm_manager.py    # 算法管理器
├── algorithms/          # 优化算法实现
│   ├── base_optimizer.py       # 算法基类
│   ├── ppo_optimizer.py        # PPO强化学习
│   ├── simulated_annealing.py  # 模拟退火
│   ├── genetic_algorithm.py    # 遗传算法
│   └── constraint_manager.py   # 约束管理器
├── comparison/          # 结果对比分析
│   └── results_comparator.py   # 结果对比器
├── network/             # 网络生成核心
│   ├── network.py              # 单层网络生成
│   ├── super_network.py        # 多层网络管理
│   ├── node_creators.py        # 节点创建策略
│   ├── node.py                # 节点定义
│   └── graph_manager.py        # 图管理器
├── rl_optimizer/        # 强化学习组件
│   ├── env/                    # 环境定义
│   │   ├── layout_env.py       # 布局优化环境
│   │   ├── cost_calculator.py  # 成本计算器
│   │   └── vec_env_wrapper.py  # 矢量化环境包装器
│   ├── model/                  # 模型定义
│   │   └── policy_network.py   # Transformer策略网络
│   ├── utils/                  # 工具函数
│   │   ├── lr_scheduler.py     # 学习率调度器
│   │   └── checkpoint_callback.py # 检查点回调
│   └── data/                   # 数据管理
│       └── cache_manager.py    # 缓存管理器
├── analysis/            # 分析工具
│   ├── travel_time.py          # 行程时间计算
│   ├── process_flow.py         # 流程分析
│   └── word_detect.py          # 词汇检测
├── image_processing/    # 图像处理
│   └── processor.py            # 图像预处理
└── plotting/            # 可视化工具
    └── plotter.py              # 图表绘制
```

## 🛠️ 技术栈

- **Python 3.11+** - 现代Python开发环境
- **PyTorch** - 深度学习框架
- **Stable-baselines3** - 强化学习算法库
- **OpenCV + PIL** - 图像处理
- **NetworkX** - 图结构分析
- **Plotly** - 交互式可视化
- **Pandas + NumPy** - 数据分析
- **uv** - 现代Python包管理器

## 🚀 快速开始

### 环境要求
- Python 3.11+
- CUDA 11.8+ (可选，用于GPU加速)
- 8GB+ RAM

### 安装依赖
```bash
# 克隆项目
git clone <repository-url>
cd Hospital_Room_Dectection

# 安装uv（如果尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 同步依赖
uv sync
```

### 数据准备
确保数据目录结构如下：
```
data/
├── image/          # 楼层平面原图
│   ├── -1F.png
│   ├── 1F.png
│   ├── 2F.png
│   └── ...
└── label/          # 楼层标注图（颜色映射）
    ├── -1F-meng.png
    ├── 1F-meng.png  
    ├── 2F-meng.png
    └── ...
```

## 📖 使用指南

### 1. 网络生成
从楼层平面图生成多层网络：
```bash
# 基础网络生成
uv run python main.py --mode network

# 自定义图像目录
uv run python main.py --mode network --image-dir ./custom/label/
```

输出文件：
- `results/network/hospital_network_3d.html` - 交互式3D网络图
- `results/network/hospital_travel_times.csv` - 行程时间矩阵

### 2. 单算法优化
运行特定优化算法：
```bash
# PPO强化学习
uv run python main.py --mode optimize --algorithm ppo --total-timesteps 100000

# 模拟退火
uv run python main.py --mode optimize --algorithm simulated_annealing \
  --initial-temperature 1500.0 --max-iterations 5000

# 遗传算法  
uv run python main.py --mode optimize --algorithm genetic_algorithm \
  --population-size 50 --max-iterations 300
```

#### PPO训练特性
PPO算法支持以下高级功能：
- **自动保存最佳模型**：训练过程中通过EvalCallback自动评估并保存最佳模型
- **断点续训**：支持从checkpoint恢复训练，避免长时间训练丢失进度
- **学习率调度**：支持线性衰减学习率，提高训练稳定性和收敛性
- **动作掩码**：使用MaskablePPO确保只选择有效动作，避免违反约束

### 3. PPO模型推理
使用已训练的PPO模型进行推理：
```bash
# 默认使用最新训练的最佳模型
uv run python inference_ppo_model.py

# 指定模型文件
uv run python inference_ppo_model.py \
  --model-path results/model/ppo_layout_20250810-173909/best_model/best_model.zip

# 多次推理取最佳
uv run python inference_ppo_model.py --n-episodes 10

# 单次推理并显示详细过程
uv run python inference_ppo_model.py --single --verbose
```

推理输出：
- `results/inference/ppo_inference_*.json` - 详细推理结果
- `results/inference/layout_ppo_inference_*.txt` - 布局方案文本格式

### 4. 多算法对比
运行多个算法进行性能对比：
```bash
# 基础对比
uv run python main.py --mode compare \
  --algorithms ppo,simulated_annealing,genetic_algorithm

# 并行执行（需要充足系统资源）
uv run python main.py --mode compare \
  --algorithms simulated_annealing,genetic_algorithm --parallel

# 快速对比（不生成图表）
uv run python main.py --mode compare \
  --algorithms ppo,simulated_annealing --no-plots --no-report
```

## 🎛️ 配置参数

### 网络生成配置
在`src/config.py`的`NetworkConfig`中配置：
- `AREA_THRESHOLD = 60` - 节点最小面积阈值
- `GRID_SIZE = 40` - 网格节点大小
- `IMAGE_ROTATE = 180` - 图像旋转角度

### 优化算法配置
在`src/config.py`的`RLConfig`中配置：

**约束参数：**
- `AREA_SCALING_FACTOR = 0.1` - 面积约束容差
- `MANDATORY_ADJACENCY` - 强制相邻约束
- `PREFERRED_ADJACENCY` - 偏好相邻约束
- `FIXED_NODE_TYPES` - 固定位置的节点类型

**PPO训练参数：**
- `TOTAL_TIMESTEPS = 5_000_000` - 总训练步数
- `NUM_ENVS = 8` - 并行环境数
- `EVAL_FREQUENCY = 10000` - 评估频率
- `CHECKPOINT_FREQUENCY = 50000` - 检查点保存频率
- `RESUME_TRAINING = False` - 是否启用断点续训
- `LEARNING_RATE_SCHEDULE_TYPE = "linear"` - 学习率调度类型

**奖励函数参数：**
- `ENABLE_POTENTIAL_REWARD = True` - 启用势函数奖励
- `AREA_MATCH_REWARD_WEIGHT = 0.2` - 面积匹配奖励权重
- `REWARD_PLACEMENT_BONUS = 1.0` - 成功放置奖励
- `REWARD_EMPTY_SLOT_PENALTY = 5.0` - 空槽位惩罚

## 📊 输出结果

### 目录结构
```
results/
├── network/                    # 网络生成结果
│   ├── hospital_network_3d.html
│   └── hospital_travel_times.csv
├── comparison/                 # 算法对比结果
│   ├── algorithm_comparison_*.csv
│   ├── *_result_*.json
│   └── best_layouts_*.json
├── plots/                      # 对比图表
│   ├── cost_comparison_*.png
│   ├── convergence_curves_*.png
│   └── performance_radar_*.png
└── reports/                    # 分析报告
    └── algorithm_comparison_report_*.md

logs/
├── hospital_optimizer.log     # 系统运行日志
└── ppo_layout_*/              # PPO训练日志目录
    ├── final_model.zip        # 最终模型
    ├── best_model/            # 最佳模型目录
    │   └── best_model.zip     # 评估中的最佳模型
    ├── checkpoints/           # 训练检查点
    ├── eval_logs/             # 评估日志
    └── training_config.json   # 训练配置
```

## 🎯 应用场景

- **医院新建设计** - 基于需求自动生成最优布局
- **医院改扩建** - 在现有约束下优化科室布局  
- **运营效率分析** - 量化分析布局对就医效率的影响
- **算法研究** - 对比不同优化算法在空间布局问题上的性能

## 🔧 扩展开发

### 添加新的优化算法
1. 继承`BaseOptimizer`基类
2. 实现`optimize()`方法
3. 在`AlgorithmManager`中注册新算法

### 添加新的医院区域类型
在`src/config.py`的`COLOR_MAP`中添加新的RGB颜色映射即可。

### PPO模型训练技巧

#### 断点续训
当训练意外中断时，可以从最近的检查点恢复：
```bash
# 自动查找最新检查点
uv run python main.py --mode optimize --algorithm ppo --resume

# 指定检查点路径
uv run python main.py --mode optimize --algorithm ppo \
  --resume --checkpoint-path logs/ppo_layout_*/checkpoints/rl_model_1000000_steps.zip
```

#### 调试动作掩码
PPO使用MaskablePPO确保只选择有效动作。如遇到"选择已放置科室"错误，检查：
1. 环境的`get_action_mask()`方法是否正确返回掩码
2. 模型预测时是否传递了`action_masks`参数
3. 动作空间维度是否与掩码维度一致

#### 性能优化建议
- **并行环境数**：增加`NUM_ENVS`可加速训练，但需要更多内存
- **批量大小**：调整`BATCH_SIZE`平衡训练速度和稳定性
- **学习率调度**：使用线性衰减提高后期收敛稳定性
- **评估频率**：降低`EVAL_FREQUENCY`减少训练中断，但可能错过最佳模型

### 常见问题解决

#### Q: PPO训练不收敛
- 检查奖励函数设计是否合理
- 尝试调整学习率和批量大小
- 增加训练步数
- 启用势函数奖励引导

#### Q: 推理时出现约束违反
- 确认模型使用了正确的动作掩码
- 检查约束管理器配置是否一致
- 验证训练和推理环境配置相同

## 📄 许可证

本项目采用MIT许可证 - 查看[LICENSE](LICENSE)文件了解详情。

---

**Medical Layout Optimizer** - 让医院布局设计更智能、更高效 🏥