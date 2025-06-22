# 强化学习布局优化架构设计 v1.0

## 文档信息
- **创建时间:** 2025-06-22 02:10:35 +08:00
- **版本:** 1.0
- **负责人:** AR
- **关联任务:** P3-AR-001

## 1. 架构概述

本架构设计基于Deep Q-Network (DQN)算法，为医院布局优化提供智能学习能力。系统将现有的贪心搜索优化器替换为基于强化学习的解决方案。

### 1.1 核心组件
- **DQNAgent**: 深度Q网络智能体
- **ReplayBuffer**: 经验回放缓冲区
- **RLEnvironment**: 强化学习环境包装器
- **StateEncoder**: 状态编码器
- **RLLayoutOptimizer**: 强化学习布局优化器

## 2. 状态空间设计

### 2.1 状态表示方案
状态空间包含当前的功能分配信息，编码为固定维度的向量：

```python
State = {
    "assignment_vector": [N_locations × N_functions] 矩阵展平,
    "workflow_features": [N_workflows × feature_dim],
    "location_features": [N_locations × location_feature_dim]
}
```

### 2.2 状态编码细节
1. **Assignment Vector**: 使用one-hot编码表示每个物理位置的当前功能分配
2. **Workflow Features**: 包含工作流的权重、序列长度等特征
3. **Location Features**: 包含位置的面积、类型、连通性等特征

### 2.3 状态维度控制
- 使用哈希技术或特征降维避免维度爆炸
- 状态向量总维度控制在1000以内，保证训练效率

## 3. 动作空间设计

### 3.1 动作定义
动作空间为离散空间，每个动作表示一次房间功能交换操作：

```python
Action = (location_A_id, location_B_id)
```

### 3.2 动作空间大小
- 最大动作数量: C(N_swappable_locations, 2)
- 使用动作掩码避免无效动作（如交换相同功能的房间）

## 4. 奖励函数设计

### 4.1 主要奖励
```python
primary_reward = -(weighted_total_travel_time)
```

### 4.2 辅助奖励
- **改进奖励**: 当总时间减少时给予额外正奖励
- **探索奖励**: 鼓励尝试新的状态配置
- **稳定性奖励**: 避免频繁无效交换

### 4.3 奖励归一化
对奖励进行标准化处理，避免数值范围过大影响学习稳定性。

## 5. 网络架构

### 5.1 DQN网络结构
```
Input Layer (state_dim) 
    ↓
Hidden Layer 1 (512 neurons, ReLU)
    ↓  
Hidden Layer 2 (256 neurons, ReLU)
    ↓
Hidden Layer 3 (128 neurons, ReLU)
    ↓
Output Layer (action_dim)
```

### 5.2 关键技术
- **Double DQN**: 减少过估计问题
- **Dueling DQN**: 分离状态价值和动作优势
- **Target Network**: 提高训练稳定性

## 6. 训练策略

### 6.1 超参数设置
- Learning Rate: 1e-4
- Batch Size: 32
- Replay Buffer Size: 50000
- Target Network Update Frequency: 1000 steps
- Epsilon Decay: 0.995 (from 1.0 to 0.01)

### 6.2 训练阶段
1. **预热阶段**: 收集初始经验
2. **探索阶段**: 高epsilon随机探索
3. **优化阶段**: 逐步降低epsilon，重点利用

## 7. 系统集成

### 7.1 接口兼容性
新的RLLayoutOptimizer将实现与现有LayoutOptimizer相同的接口：

```python
def run_optimization(self, 
                    initial_assignment: FunctionalAssignment,
                    max_iterations: int = 100) -> Tuple[FunctionalAssignment, float, List[EvaluatedWorkflowOutcome]]
```

### 7.2 性能监控
- 训练损失曲线
- 奖励变化趋势  
- 探索率衰减
- 最优解质量跟踪

## 8. 安全与稳定性设计

### 8.1 梯度裁剪
防止梯度爆炸，设置梯度裁剪阈值为1.0。

### 8.2 经验回放优先级
实现优先级经验回放，重点学习重要转换。

### 8.3 训练检查点
定期保存模型权重，支持断点续训。

## Update Log
- 2025-06-22 02:10:35 +08:00: 初始架构设计完成 (v1.0)