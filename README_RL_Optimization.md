# 基于强化学习的医院布局优化系统

本模块为医院布局优化提供了一个基于强化学习的解耦方案，可以根据 `result/super_network_travel_times.csv` 中的行程时间数据来优化功能区域的分配。

## 功能特点

- **解耦设计**: 与现有代码完全独立，可单独运行
- **多空间支持**: 支持单个功能（如门诊挂号）分配到多个物理空间
- **工作流优化**: 基于常见的医院工作流模式进行优化
- **强化学习**: 使用Q-Learning算法学习最优布局分配
- **可扩展**: 易于添加新的工作流模式和优化目标

## 核心组件

### 1. LayoutEnvironment (布局环境)
- 管理状态空间和动作空间
- 基于行程时间数据计算奖励
- 支持多种工作流模式

### 2. QLearningAgent (Q学习智能体)
- 实现Q-Learning算法
- 使用ε-贪婪策略进行探索和利用
- 支持模型保存和加载

### 3. LayoutOptimizer (布局优化器)
- 主要的优化接口
- 集成环境和智能体
- 提供训练和优化功能

## 使用方法

### 基本使用

```python
from src.analysis.rl_layout_optimizer import LayoutOptimizer, create_default_workflow_patterns

# 创建工作流模式
workflow_patterns = create_default_workflow_patterns()

# 初始化优化器
optimizer = LayoutOptimizer("result/super_network_travel_times.csv", workflow_patterns)

# 评估当前布局
current_eval = optimizer.evaluate_current_layout()
print(f"当前布局奖励: {current_eval['current_reward']:.2f}")

# 训练智能体
training_stats = optimizer.train(num_episodes=500)

# 优化布局
best_state, best_reward = optimizer.optimize_layout()
print(f"优化后奖励: {best_reward:.2f}")

# 保存结果
optimizer.save_model("result/rl_layout_model.json")
optimizer.export_optimized_layout(best_state, "result/optimized_layout.json")
```

### 运行演示

```bash
cd Hospital_Room_Dectection
python examples/rl_optimization_demo.py
```

## 工作流模式

系统支持多种医院工作流模式，例如：

- **妇科就诊**: 门 → 挂号收费 → 妇科 → 门
- **血液检查**: 门 → 挂号收费 → 采血处 → 检验中心 → 门  
- **超声检查**: 门 → 挂号收费 → 超声科 → 门
- **内科就诊**: 门 → 挂号收费 → 内科 → 内诊药房 → 门
- **急诊**: 门 → 急诊科 → 门

可以通过 `add_workflow_pattern()` 方法添加自定义工作流。

## 优化目标

系统的优化目标是最小化所有工作流模式的总行程时间。奖励函数考虑：

1. **行程时间**: 工作流中相邻步骤之间的最短行程时间
2. **多空间选择**: 当功能有多个物理空间时，选择最优路径
3. **未分配惩罚**: 对未分配功能给予高惩罚

## 输出文件

### 1. 训练模型 (`rl_layout_model.json`)
包含训练好的Q表和超参数

### 2. 优化布局 (`optimized_layout.json`)
包含优化后的功能-空间分配方案：
```json
{
  "function_to_spaces": {
    "挂号收费": ["挂号收费_10002", "挂号收费_20001"],
    "妇科": ["妇科_30005"]
  },
  "space_to_function": {
    "挂号收费_10002": "挂号收费",
    "挂号收费_20001": "挂号收费",
    "妇科_30005": "妇科"
  }
}
```

## 参数调整

### Q-Learning参数
- `learning_rate`: 学习率 (默认: 0.1)
- `discount_factor`: 折扣因子 (默认: 0.95)
- `epsilon`: 探索率 (默认: 0.1)
- `epsilon_decay`: 探索率衰减 (默认: 0.995)

### 训练参数
- `num_episodes`: 训练轮数 (默认: 1000)
- `max_steps_per_episode`: 每轮最大步数 (默认: 100)

## 扩展功能

### 添加自定义工作流
```python
# 添加复杂的工作流模式
optimizer.add_workflow_pattern(['门', '挂号收费', '妇科', '采血处', '检验中心', '门'])
```

### 自定义奖励函数
可以继承 `LayoutEnvironment` 类并重写 `_calculate_reward()` 方法来实现自定义的优化目标。

### 不同的RL算法
可以继承 `RLAgent` 基类来实现其他强化学习算法，如Deep Q-Network (DQN)、Actor-Critic等。

## 性能考虑

- 训练时间取决于状态空间大小和训练轮数
- 对于大型医院布局，建议使用更高效的RL算法
- 可以通过并行化来加速训练过程

## 依赖项

- pandas: 数据处理
- numpy: 数值计算  
- logging: 日志记录
- json: 数据序列化
- pathlib: 路径处理

所有依赖项都是Python标准库或项目已有的依赖。
