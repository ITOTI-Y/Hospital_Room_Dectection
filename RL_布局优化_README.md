# 强化学习布局优化系统

本项目为医院房间检测项目提供了基于深度强化学习(DQN)的布局优化解决方案，替代传统的贪心+随机搜索方法。

## 🎯 项目目标

通过智能学习策略优化医院房间功能分配，最小化患者就诊流程的总旅行时间，提升医院运营效率。

## ✨ 核心特性

- **先进算法**: 基于Dueling Double DQN的强化学习优化器
- **智能编码**: 定制化状态表示，有效处理复杂布局问题
- **动态奖励**: 基于改进幅度的自适应奖励函数
- **无缝集成**: 与现有系统完全兼容的接口设计
- **高度可配置**: 丰富的超参数设置，适应不同场景

## 🏗️ 系统架构

```
强化学习布局优化系统
├── DQNAgent (智能体)
│   ├── DuelingDQN (神经网络)
│   ├── ReplayBuffer (经验回放)
│   └── Target Network (目标网络)
├── RLEnvironment (环境)
│   ├── StateEncoder (状态编码)
│   ├── Action Space (动作空间)
│   └── Reward Function (奖励函数)
└── RLLayoutOptimizer (优化器)
    ├── Training Loop (训练循环)
    ├── Model Persistence (模型持久化)
    └── Performance Monitoring (性能监控)
```

## 🚀 快速开始

### 1. 环境准备

确保已安装必要依赖：
```bash
# 安装项目依赖
uv sync

# 或使用pip
pip install torch numpy pandas scikit-learn matplotlib plotly
```

### 2. 数据准备

首先运行主程序生成旅行时间数据：
```bash
python main.py
```

这将生成 `result/super_network_travel_times.csv` 文件。

### 3. 运行强化学习优化

```bash
# 运行演示脚本
python example_rl_optimization.py

# 或直接在代码中使用
from src.optimization.rl_optimizer import RLLayoutOptimizer, RLConfig

# 配置参数
rl_config = RLConfig(
    learning_rate=1e-4,
    batch_size=32,
    replay_buffer_size=50000,
    max_iterations=1000
)

# 创建优化器
optimizer = RLLayoutOptimizer(
    path_finder=path_finder,
    workflow_definitions=workflows,
    config=config,
    rl_config=rl_config
)

# 运行优化
best_assignment, best_objective, outcomes = optimizer.run_optimization(
    initial_assignment=initial_assignment,
    max_iterations=1000,
    save_model_path="model.pth"
)
```

### 4. 结果分析

优化完成后，系统会自动生成：
- `result/rl_layout_model.pth`: 训练好的模型
- `result/rl_optimization_results.json`: 优化结果汇总
- `logs/rl_optimization.log`: 详细训练日志

## ⚙️ 配置说明

### RLConfig 参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| learning_rate | 1e-4 | 学习率 |
| batch_size | 32 | 批次大小 |
| replay_buffer_size | 50000 | 经验回放缓冲区大小 |
| target_update_frequency | 1000 | 目标网络更新频率 |
| epsilon_start | 1.0 | 初始探索率 |
| epsilon_end | 0.01 | 最终探索率 |
| epsilon_decay | 0.995 | 探索率衰减 |
| gamma | 0.99 | 折扣因子 |

### 网络结构参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| hidden_dim1 | 512 | 第一隐藏层维度 |
| hidden_dim2 | 256 | 第二隐藏层维度 |
| hidden_dim3 | 128 | 第三隐藏层维度 |
| gradient_clip | 1.0 | 梯度裁剪阈值 |

## 📊 性能监控

系统提供实时监控功能：

```python
# 训练过程中会输出：
# Episode 100: Avg Reward = 15.32, Epsilon = 0.905, Best Objective = 1250.45
# Episode 200: Avg Reward = 25.67, Epsilon = 0.819, Best Objective = 1180.23
```

监控指标包括：
- 平均奖励变化
- 探索率衰减
- 最优目标值跟踪
- 训练损失曲线

## 🧪 测试运行

运行测试套件验证系统功能：

```bash
# 运行所有测试
python project_document/tests/e2e/scripts/test_rl_optimizer.py

# 运行特定测试
python -m unittest project_document.tests.e2e.scripts.test_rl_optimizer.TestStateEncoder
```

## 🎛️ 高级用法

### 自定义奖励函数

```python
class CustomRLEnvironment(RLEnvironment):
    def _calculate_reward(self, old_objective, new_objective):
        # 自定义奖励逻辑
        improvement = old_objective - new_objective
        # 添加额外的约束奖励
        constraint_penalty = self._calculate_constraint_penalty()
        return improvement * 100 - constraint_penalty
```

### 模型加载和继续训练

```python
# 加载预训练模型
agent = DQNAgent(state_dim, action_dim, config)
agent.load("pretrained_model.pth")

# 继续训练
optimizer.run_optimization(
    initial_assignment=assignment,
    max_iterations=500,  # 额外训练轮次
    save_model_path="continued_model.pth"
)
```

### 超参数调优

```python
# 网格搜索示例
learning_rates = [1e-5, 1e-4, 1e-3]
batch_sizes = [16, 32, 64]

best_config = None
best_performance = float('inf')

for lr in learning_rates:
    for bs in batch_sizes:
        config = RLConfig(learning_rate=lr, batch_size=bs)
        # 运行优化并评估性能
        result = evaluate_config(config)
        if result < best_performance:
            best_performance = result
            best_config = config
```

## 🔧 故障排除

### 常见问题

1. **CUDA内存不足**
   ```python
   # 使用CPU训练
   rl_config = RLConfig(device="cpu")
   ```

2. **收敛速度慢**
   ```python
   # 增加学习率或减小网络大小
   rl_config = RLConfig(
       learning_rate=5e-4,
       hidden_dim1=256,
       hidden_dim2=128
   )
   ```

3. **状态空间过大**
   ```python
   # 减少位置数量或简化特征编码
   # 可以通过过滤不重要的位置来减小状态空间
   ```

### 性能优化建议

- 使用GPU加速训练 (如果可用)
- 合理设置经验回放缓冲区大小
- 根据问题规模调整网络架构
- 使用早停策略避免过训练

## 📈 与传统方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| 传统贪心搜索 | 简单快速 | 容易陷入局部最优 | 小规模问题 |
| 强化学习DQN | 全局优化能力强 | 训练时间较长 | 复杂大规模问题 |

## 📝 项目文档

详细文档位于 `project_document/` 目录：
- `强化学习布局优化任务.md`: 完整项目记录
- `architecture/rl_optimization_arch_v1.0.md`: 架构设计文档
- `tests/e2e/scripts/`: 测试脚本

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进项目：

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送分支 (`git push origin feature/amazing-feature`)
5. 创建Pull Request

## 📄 许可证

本项目使用MIT许可证。详见 LICENSE 文件。

---

**开发团队**: Sun Wukong (孙悟空) AI 编程助手
**项目状态**: ✅ 生产就绪
**最后更新**: 2025-06-22