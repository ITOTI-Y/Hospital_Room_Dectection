# 基于强化学习的医院布局优化系统

本模块为医院布局优化提供了一个基于强化学习的解耦方案，可以根据 `result/super_network_travel_times.csv` 中的行程时间数据来优化功能区域的分配。

## 🚀 系统概述

该系统通过强化学习算法自动优化医院各功能区域的空间分配，以最小化患者和医护人员的总行程时间。系统支持复杂的医院工作流模式，能够处理单个功能分配到多个物理空间的情况，并提供CPU和GPU两种实现方案。

## ✨ 核心功能特点

### 🔧 技术特性
- **解耦设计**: 与现有代码完全独立，可单独运行，不影响原有系统
- **双重实现**: 提供CPU版本(Q-Learning)和GPU版本(Deep Q-Network)
- **智能硬件检测**: 自动检测GPU可用性，无缝回退到CPU执行
- **多空间支持**: 支持单个功能（如门诊挂号）分配到多个物理空间
- **工作流优化**: 基于12+种常见医院工作流模式进行优化
- **模型持久化**: 支持训练模型的保存、加载和导出功能

### 🏥 医院应用特性
- **实时优化**: 根据实际行程时间数据进行布局优化
- **多科室支持**: 涵盖妇科、儿科、内科、急诊等多个科室
- **复杂路径**: 支持多步骤就诊流程（如：挂号→科室→检查→取药）
- **空间灵活性**: 允许功能区域在多个物理位置之间灵活分配
- **可扩展性**: 易于添加新的工作流模式和优化目标

## 🏗️ 系统架构

### 📊 CPU版本组件 (`rl_layout_optimizer.py`)

#### 1. LayoutEnvironment (布局环境)
**功能说明**: 强化学习的核心环境，模拟医院布局优化问题
- **状态空间管理**: 维护当前功能-空间分配状态，支持107维状态向量
- **动作空间定义**: 管理5412种可能的分配动作（功能到空间的映射）
- **奖励计算**: 基于行程时间数据计算奖励，奖励越高表示布局越优
- **工作流支持**: 支持12+种医院工作流模式的同时优化
- **约束处理**: 确保每个功能至少分配到一个空间，避免无效状态

#### 2. QLearningAgent (Q学习智能体)
**功能说明**: 基于表格的Q-Learning算法实现
- **Q表管理**: 维护状态-动作价值表，记录每个状态下各动作的期望回报
- **ε-贪婪策略**: 平衡探索和利用，ε=0.1时有10%概率随机探索
- **学习更新**: 使用Bellman方程更新Q值，学习率α=0.1，折扣因子γ=0.95
- **模型持久化**: 支持Q表的JSON格式保存和加载
- **收敛监控**: 跟踪学习进度，支持ε衰减策略

#### 3. LayoutOptimizer (布局优化器)
**功能说明**: 系统的主要接口，整合环境和智能体
- **数据加载**: 自动解析`super_network_travel_times.csv`文件
- **训练管理**: 控制训练过程，支持自定义训练轮数和步数
- **优化执行**: 使用训练好的模型寻找最优布局分配
- **结果导出**: 生成优化后的布局配置文件
- **性能评估**: 提供当前布局和优化布局的性能对比

### 🚀 GPU版本组件 (`rl_layout_optimizer_gpu.py`)

#### 1. DQNNetwork (深度Q网络)
**功能说明**: 基于PyTorch的神经网络实现
- **网络架构**: 三层全连接网络（输入层→512隐藏层→输出层）
- **激活函数**: 使用ReLU激活函数，提供非线性表达能力
- **GPU加速**: 支持CUDA加速计算，自动检测GPU可用性
- **批处理**: 支持批量数据处理，提高训练效率
- **梯度优化**: 使用Adam优化器，学习率1e-3

#### 2. DQNAgent (深度Q学习智能体)
**功能说明**: 基于神经网络的强化学习智能体
- **经验回放**: 维护经验缓冲区，支持批量学习
- **目标网络**: 使用目标网络稳定训练过程
- **ε-贪婪策略**: 动态调整探索率，支持ε衰减
- **损失计算**: 使用均方误差损失函数
- **内存管理**: 智能管理GPU内存，避免内存溢出

#### 3. GPULayoutEnvironment (GPU布局环境)
**功能说明**: GPU优化的环境实现
- **张量计算**: 所有计算使用PyTorch张量，支持GPU加速
- **批量处理**: 支持批量状态和动作处理
- **内存优化**: 优化张量创建和转换过程
- **设备管理**: 自动管理CPU/GPU设备切换
- **性能监控**: 提供GPU内存使用情况监控

#### 4. GPULayoutOptimizer (GPU布局优化器)
**功能说明**: GPU版本的主要接口
- **硬件检测**: 自动检测CUDA可用性和GPU信息
- **智能回退**: CUDA不可用时自动使用CPU执行
- **加速训练**: GPU环境下显著提升训练速度
- **模型保存**: 支持PyTorch模型格式(.pth)的保存和加载
- **性能对比**: 提供CPU和GPU版本的性能对比数据

## 📖 详细使用指南

### 🔰 快速开始

#### 方法一：运行演示脚本（推荐新手）

```bash
# 进入项目目录
cd Hospital_Room_Dectection

# 运行CPU版本演示（适合所有环境）
python examples/rl_optimization_demo.py

# 运行GPU版本演示（需要CUDA支持）
python examples/gpu_optimization_demo.py

# 测试GPU兼容性
python test_gpu_compatibility.py
```

#### 方法二：编程接口使用

##### CPU版本使用示例
```python
from src.analysis.rl_layout_optimizer import LayoutOptimizer, create_default_workflow_patterns

# 1. 创建工作流模式
workflow_patterns = create_default_workflow_patterns()
print(f"加载了 {len(workflow_patterns)} 个工作流模式")

# 2. 初始化优化器
optimizer = LayoutOptimizer("result/super_network_travel_times.csv", workflow_patterns)
print("CPU优化器初始化完成")

# 3. 评估当前布局
current_eval = optimizer.evaluate_current_layout()
print(f"当前布局奖励: {current_eval['current_reward']:.2f}")
print(f"发现 {len(current_eval['workflow_penalties'])} 个工作流")

# 4. 训练智能体（Q-Learning）
print("开始训练Q-Learning智能体...")
training_stats = optimizer.train(num_episodes=500, max_steps_per_episode=100)
print(f"训练完成，最终探索率: {training_stats['final_epsilon']:.3f}")

# 5. 优化布局
print("开始布局优化...")
best_state, best_reward = optimizer.optimize_layout(max_iterations=200)
print(f"优化后奖励: {best_reward:.2f}")
print(f"改进幅度: {best_reward - current_eval['current_reward']:.2f}")

# 6. 保存结果
optimizer.save_model("result/rl_layout_model.json")
optimizer.export_optimized_layout(best_state, "result/optimized_layout.json")
print("结果已保存")
```

##### GPU版本使用示例
```python
from src.analysis.rl_layout_optimizer_gpu import GPULayoutOptimizer, create_default_workflow_patterns, check_gpu_availability

# 1. 检查GPU可用性
gpu_info = check_gpu_availability()
print("GPU信息:")
for key, value in gpu_info.items():
    print(f"  {key}: {value}")

# 2. 创建工作流模式
workflow_patterns = create_default_workflow_patterns()

# 3. 初始化GPU优化器
optimizer = GPULayoutOptimizer("result/super_network_travel_times.csv", workflow_patterns)
print("GPU优化器初始化完成")

# 4. 评估当前布局
current_eval = optimizer.evaluate_current_layout()
print(f"当前布局奖励: {current_eval['current_reward']:.2f}")
print(f"使用设备: {current_eval['device_used']}")

# 5. 训练DQN智能体
print("开始训练Deep Q-Network...")
training_stats = optimizer.train(num_episodes=500, max_steps_per_episode=50)
print(f"训练完成，平均损失: {sum(training_stats['losses'][-100:]) / 100:.4f}")

# 6. 优化布局
best_state, best_reward = optimizer.optimize_layout(max_iterations=200)
print(f"最优奖励: {best_reward:.2f}")

# 7. 保存GPU模型
optimizer.save_model("result/rl_layout_model_gpu.pth")
optimizer.export_optimized_layout(best_state, "result/optimized_layout_gpu.json")
print("GPU模型和结果已保存")
```

### 🔧 高级配置

#### 自定义工作流模式
```python
# 添加自定义工作流
optimizer.add_workflow_pattern(['门', '挂号收费', '妇科', '超声科', '门'])
optimizer.add_workflow_pattern(['门', '急诊科', '放射科', '门'])

# 批量添加复杂工作流
complex_workflows = [
    ['门', '挂号收费', '内科', '采血处', '检验中心', '内诊药房', '门'],
    ['门', '挂号收费', '儿科', '超声科', '采血处', '门'],
    ['门', '挂号收费', '妇科', '放射科', '超声科', '门']
]
for workflow in complex_workflows:
    optimizer.add_workflow_pattern(workflow)
```

#### 训练参数调优
```python
# CPU版本参数调优
training_stats = optimizer.train(
    num_episodes=1000,        # 训练轮数
    max_steps_per_episode=150, # 每轮最大步数
    learning_rate=0.1,        # 学习率
    discount_factor=0.95,     # 折扣因子
    epsilon=0.1,              # 初始探索率
    epsilon_decay=0.995       # 探索率衰减
)

# GPU版本参数调优
training_stats = gpu_optimizer.train(
    num_episodes=500,         # 训练轮数
    max_steps_per_episode=50, # 每轮最大步数
    batch_size=32,           # 批处理大小
    learning_rate=1e-3,      # 神经网络学习率
    memory_size=10000,       # 经验回放缓冲区大小
    target_update=10         # 目标网络更新频率
)
```

## 🏥 医院工作流模式详解

### 📋 预定义工作流模式

系统内置了12种常见的医院工作流模式，涵盖了大部分医院科室和就诊流程：

#### 🩺 基础就诊流程
1. **妇科就诊**: `门 → 挂号收费 → 妇科 → 门`
   - 适用场景：常规妇科检查、咨询
   - 预期时间：短程就诊流程
   - 优化重点：挂号收费与妇科的距离

2. **儿科就诊**: `门 → 挂号收费 → 儿科 → 门`
   - 适用场景：儿童常规检查、疫苗接种
   - 特殊考虑：儿科通常需要便于家长携带儿童

3. **眼科就诊**: `门 → 挂号收费 → 眼科 → 门`
   - 适用场景：视力检查、眼部疾病诊治
   - 优化重点：减少等待时间和行走距离

#### 🔬 检查检验流程
4. **血液检查**: `门 → 挂号收费 → 采血处 → 检验中心 → 门`
   - 适用场景：血常规、生化检查
   - 关键路径：采血处到检验中心的便捷性
   - 时间敏感：血样需要及时送检

5. **超声检查**: `门 → 挂号收费 → 超声科 → 门`
   - 适用场景：腹部、心脏、妇科超声
   - 设备特点：超声设备相对固定，位置优化重要

6. **放射检查**: `门 → 挂号收费 → 放射科 → 门`
   - 适用场景：X光、CT、MRI检查
   - 安全考虑：放射科通常位置相对独立

#### 🏥 综合诊疗流程
7. **内科就诊**: `门 → 挂号收费 → 内科 → 内诊药房 → 门`
   - 适用场景：内科疾病诊治和取药
   - 完整流程：诊断→开药→取药
   - 优化重点：内科与药房的连接

8. **急诊流程**: `门 → 急诊科 → 门`
   - 适用场景：紧急医疗情况
   - 时间关键：最短路径至关重要
   - 特殊要求：24小时可达性

#### 🔄 复杂多步流程
9. **妇科综合检查**: `门 → 挂号收费 → 妇科 → 采血处 → 检验中心 → 门`
   - 适用场景：妇科疾病诊断需要血液检查
   - 多科室协作：妇科→检验科
   - 时间安排：需要合理的检查顺序

10. **超声+妇科**: `门 → 挂号收费 → 超声科 → 妇科 → 门`
    - 适用场景：妇科疾病需要超声确诊
    - 检查顺序：超声检查→医生诊断
    - 路径优化：两个科室间的便捷连接

11. **内科+影像**: `门 → 挂号收费 → 内科 → 放射科 → 内诊药房 → 门`
    - 适用场景：内科疾病需要影像学检查
    - 复杂流程：诊断→检查→再诊断→取药
    - 多次往返：需要优化科室间距离

12. **儿科+检验**: `门 → 挂号收费 → 儿科 → 采血处 → 门`
    - 适用场景：儿童疾病需要血液检查
    - 特殊考虑：儿童采血的便利性和安全性

### 🎯 工作流优化策略

#### 优化目标
- **最小化总行程时间**: 减少患者和医护人员的移动时间
- **平衡科室负载**: 避免某些区域过度拥挤
- **提高就诊效率**: 优化科室间的连接性
- **考虑特殊需求**: 急诊、儿科等特殊科室的位置要求

#### 权重分配
系统根据工作流的使用频率和重要性分配不同权重：
- **急诊流程**: 最高权重，优先保证最短路径
- **常规就诊**: 中等权重，平衡效率和资源利用
- **复杂检查**: 较低权重，但考虑检查流程的合理性

### 🔧 自定义工作流

#### 添加新工作流
```python
# 添加单个工作流
optimizer.add_workflow_pattern(['门', '挂号收费', '肿瘤科', '放射科', '门'])

# 添加多个相关工作流
cancer_workflows = [
    ['门', '挂号收费', '肿瘤科', '门'],
    ['门', '挂号收费', '肿瘤科', '放射科', '门'],
    ['门', '挂号收费', '肿瘤科', '采血处', '检验中心', '门']
]
for workflow in cancer_workflows:
    optimizer.add_workflow_pattern(workflow)
```

#### 工作流设计原则
1. **起点终点**: 所有工作流都应以"门"开始和结束
2. **逻辑顺序**: 遵循实际就诊的逻辑顺序
3. **必要步骤**: 包含所有必要的就诊步骤
4. **现实可行**: 确保工作流在实际医院中可行

#### 工作流验证
系统会自动验证新添加的工作流：
- 检查所有功能是否在数据中存在
- 验证工作流的连通性
- 计算工作流的可行性评分

## 🎯 优化目标与算法原理

### 📊 优化目标详解

系统的核心优化目标是**最小化所有工作流模式的总行程时间**，同时确保医院布局的合理性和可行性。

#### 主要优化指标

1. **总行程时间最小化**
   - **计算方法**: 对所有工作流模式中相邻步骤间的行程时间求和
   - **数据来源**: 基于`super_network_travel_times.csv`中的实际测量数据
   - **权重考虑**: 不同工作流根据使用频率分配不同权重
   - **示例**: 妇科就诊流程的总时间 = 门到挂号收费时间 + 挂号收费到妇科时间 + 妇科到门时间

2. **多空间智能选择**
   - **问题描述**: 单个功能（如挂号收费）可能有多个物理位置
   - **选择策略**: 对每个工作流步骤，自动选择距离最近的可用空间
   - **动态优化**: 考虑前后步骤的位置，选择全局最优路径
   - **示例**: 如果有3个挂号收费点，系统会为每个患者选择最近的那个

3. **布局完整性保证**
   - **分配约束**: 确保每个功能至少分配到一个物理空间
   - **连通性检查**: 验证所有分配的空间在网络中可达
   - **容量考虑**: 避免某些空间过度分配（未来扩展）

### 🧮 奖励函数设计

#### CPU版本奖励函数（Q-Learning）
```python
def calculate_reward(self, state):
    """
    计算当前状态的奖励值
    奖励 = -总行程时间 - 未分配惩罚
    """
    total_penalty = 0
    
    # 1. 计算所有工作流的行程时间惩罚
    for workflow_pattern in self.workflow_patterns:
        workflow_penalty = 0
        for i in range(len(workflow_pattern) - 1):
            current_func = workflow_pattern[i]
            next_func = workflow_pattern[i + 1]
            
            # 获取当前功能的所有可用空间
            current_spaces = state.function_to_spaces.get(current_func, [])
            next_spaces = state.function_to_spaces.get(next_func, [])
            
            if not current_spaces or not next_spaces:
                # 未分配惩罚：10000单位
                workflow_penalty += 10000
            else:
                # 选择最短路径
                min_time = float('inf')
                for curr_space in current_spaces:
                    for next_space in next_spaces:
                        travel_time = self.get_travel_time(curr_space, next_space)
                        min_time = min(min_time, travel_time)
                workflow_penalty += min_time
        
        total_penalty += workflow_penalty
    
    # 2. 未分配功能的额外惩罚
    for function in self.all_functions:
        if not state.function_to_spaces.get(function):
            total_penalty += 10000  # 高惩罚确保所有功能都被分配
    
    return -total_penalty  # 负值表示惩罚，越小越好
```

#### GPU版本奖励函数（DQN）
```python
def calculate_reward_tensor(self, state_tensor):
    """
    GPU加速的奖励计算，使用PyTorch张量操作
    """
    # 使用张量运算并行计算所有工作流的奖励
    batch_size = state_tensor.size(0)
    total_rewards = torch.zeros(batch_size, device=self.device)
    
    for workflow_idx, workflow in enumerate(self.workflow_patterns):
        # 并行计算批量状态的工作流奖励
        workflow_rewards = self._calculate_workflow_reward_batch(
            state_tensor, workflow
        )
        total_rewards += workflow_rewards
    
    return total_rewards
```

### 🔄 强化学习算法原理

#### CPU版本：Q-Learning算法

**算法特点**:
- **无模型学习**: 不需要环境的先验知识
- **表格存储**: 使用Q表存储状态-动作价值
- **收敛保证**: 在满足条件下保证收敛到最优策略

**更新公式**:
```
Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
```
其中：
- `s`: 当前状态（布局配置）
- `a`: 动作（改变某个功能的空间分配）
- `r`: 即时奖励（布局改进带来的奖励）
- `α`: 学习率（0.1）
- `γ`: 折扣因子（0.95）
- `s'`: 下一状态

**探索策略**:
```python
def select_action(self, state):
    if random.random() < self.epsilon:
        # 探索：随机选择动作
        return random.choice(valid_actions)
    else:
        # 利用：选择Q值最高的动作
        return argmax(Q_table[state])
```

#### GPU版本：Deep Q-Network (DQN)

**算法特点**:
- **神经网络近似**: 使用深度神经网络近似Q函数
- **经验回放**: 存储历史经验，批量学习
- **目标网络**: 使用目标网络稳定训练
- **GPU加速**: 利用并行计算加速训练

**网络架构**:
```python
class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 512)      # 输入层
        self.fc2 = nn.Linear(512, 512)             # 隐藏层
        self.fc3 = nn.Linear(512, action_size)     # 输出层
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```

**训练过程**:
1. **经验收集**: 智能体与环境交互，收集(s,a,r,s')经验
2. **批量采样**: 从经验缓冲区随机采样批量数据
3. **目标计算**: 使用目标网络计算目标Q值
4. **损失计算**: 计算预测Q值与目标Q值的均方误差
5. **反向传播**: 更新主网络参数
6. **目标网络更新**: 定期更新目标网络参数

### 📈 性能评估指标

#### 训练过程监控
- **平均奖励**: 每轮训练的平均奖励值
- **损失函数**: DQN的训练损失（仅GPU版本）
- **探索率**: ε值的变化趋势
- **收敛速度**: 达到稳定性能所需的训练轮数

#### 优化结果评估
- **奖励改进**: 优化前后的奖励值对比
- **行程时间减少**: 具体的时间节省量
- **工作流效率**: 各个工作流的性能改进
- **空间利用率**: 各个物理空间的分配情况

## 📁 输出文件详解

### 🤖 CPU版本输出文件

#### 1. 训练模型文件 (`result/rl_layout_model.json`)
**文件说明**: 包含完整的Q-Learning训练结果和模型参数

**文件结构**:
```json
{
  "q_table": {
    "state_hash_1": {
      "action_1": 0.85,
      "action_2": 0.92,
      "action_3": 0.78
    },
    "state_hash_2": {
      "action_1": 0.91,
      "action_2": 0.88
    }
  },
  "hyperparameters": {
    "learning_rate": 0.1,
    "discount_factor": 0.95,
    "epsilon": 0.01,
    "epsilon_decay": 0.995
  },
  "training_info": {
    "num_episodes": 500,
    "final_epsilon": 0.01,
    "training_time": "2024-06-27 10:30:15",
    "convergence_episode": 387
  },
  "state_action_mapping": {
    "state_size": 107,
    "action_size": 5412,
    "function_count": 41,
    "space_count": 66
  }
}
```

**使用方法**:
```python
# 加载已训练的模型
optimizer = LayoutOptimizer(csv_path, workflow_patterns)
optimizer.load_model("result/rl_layout_model.json")

# 直接使用训练好的模型进行优化
best_state, best_reward = optimizer.optimize_layout()
```

#### 2. 优化布局文件 (`result/optimized_layout.json`)
**文件说明**: 包含优化后的完整功能-空间分配方案

**文件结构**:
```json
{
  "optimization_info": {
    "optimization_time": "2024-06-27 10:35:22",
    "original_reward": -72543.25,
    "optimized_reward": -68234.18,
    "improvement": 4309.07,
    "improvement_percentage": 5.94
  },
  "function_to_spaces": {
    "挂号收费": ["挂号收费_10002", "挂号收费_20001"],
    "妇科": ["妇科_30005"],
    "儿科": ["儿科_40012"],
    "采血处": ["采血处_15008", "采血处_25003"],
    "检验中心": ["检验中心_35001"],
    "超声科": ["超声科_45007"],
    "内科": ["内科_50009"],
    "内诊药房": ["内诊药房_55002"],
    "急诊科": ["急诊科_60001"],
    "眼科": ["眼科_65004"],
    "放射科": ["放射科_70005"]
  },
  "space_to_function": {
    "挂号收费_10002": "挂号收费",
    "挂号收费_20001": "挂号收费",
    "妇科_30005": "妇科",
    "儿科_40012": "儿科",
    "采血处_15008": "采血处",
    "采血处_25003": "采血处",
    "检验中心_35001": "检验中心",
    "超声科_45007": "超声科",
    "内科_50009": "内科",
    "内诊药房_55002": "内诊药房",
    "急诊科_60001": "急诊科",
    "眼科_65004": "眼科",
    "放射科_70005": "放射科"
  },
  "workflow_analysis": {
    "妇科就诊": {
      "original_time": 1148.50,
      "optimized_time": 987.25,
      "improvement": 161.25,
      "path": ["门_00001", "挂号收费_10002", "妇科_30005", "门_00001"]
    },
    "血液检查": {
      "original_time": 963.00,
      "optimized_time": 856.75,
      "improvement": 106.25,
      "path": ["门_00001", "挂号收费_20001", "采血处_15008", "检验中心_35001", "门_00001"]
    }
  },
  "statistics": {
    "total_functions": 41,
    "assigned_functions": 41,
    "total_spaces": 66,
    "utilized_spaces": 45,
    "space_utilization_rate": 68.18,
    "average_spaces_per_function": 1.12
  }
}
```

### 🚀 GPU版本输出文件

#### 3. GPU训练模型文件 (`result/rl_layout_model_gpu.pth`)
**文件说明**: PyTorch格式的深度神经网络模型文件

**文件内容**:
- 训练好的DQN网络权重
- 优化器状态
- 训练超参数
- 网络架构信息

**加载方法**:
```python
# 加载GPU训练的模型
gpu_optimizer = GPULayoutOptimizer(csv_path, workflow_patterns)
gpu_optimizer.load_model("result/rl_layout_model_gpu.pth")

# 继续训练或直接优化
best_state, best_reward = gpu_optimizer.optimize_layout()
```

#### 4. GPU优化布局文件 (`result/optimized_layout_gpu.json`)
**文件说明**: GPU版本的优化结果，格式与CPU版本相同但包含GPU特定信息

**额外信息**:
```json
{
  "gpu_info": {
    "cuda_available": true,
    "device_used": "cuda:0",
    "gpu_name": "NVIDIA RTX 4090",
    "memory_allocated": "2.34 GB",
    "training_time_gpu": "00:03:45",
    "training_time_cpu_equivalent": "00:15:20",
    "speedup_factor": 4.1
  },
  "dqn_training_stats": {
    "final_loss": 0.0234,
    "episodes_trained": 500,
    "batch_size": 32,
    "memory_size": 10000,
    "target_update_frequency": 10
  }
}
```

### 📊 结果分析工具

#### 性能对比脚本
```python
def compare_optimization_results():
    """比较CPU和GPU版本的优化结果"""
    
    # 加载CPU结果
    with open("result/optimized_layout.json", 'r') as f:
        cpu_result = json.load(f)
    
    # 加载GPU结果
    with open("result/optimized_layout_gpu.json", 'r') as f:
        gpu_result = json.load(f)
    
    print("=== 优化结果对比 ===")
    print(f"CPU版本改进: {cpu_result['optimization_info']['improvement']:.2f}")
    print(f"GPU版本改进: {gpu_result['optimization_info']['improvement']:.2f}")
    
    print("\n=== 训练效率对比 ===")
    if 'gpu_info' in gpu_result:
        gpu_time = gpu_result['gpu_info']['training_time_gpu']
        cpu_equiv_time = gpu_result['gpu_info']['training_time_cpu_equivalent']
        speedup = gpu_result['gpu_info']['speedup_factor']
        print(f"GPU训练时间: {gpu_time}")
        print(f"CPU等效时间: {cpu_equiv_time}")
        print(f"加速倍数: {speedup}x")
```

#### 可视化分析
```python
def visualize_optimization_results():
    """可视化优化结果"""
    import matplotlib.pyplot as plt
    
    # 加载结果数据
    with open("result/optimized_layout.json", 'r') as f:
        result = json.load(f)
    
    # 绘制工作流改进图
    workflows = list(result['workflow_analysis'].keys())
    improvements = [result['workflow_analysis'][w]['improvement'] 
                   for w in workflows]
    
    plt.figure(figsize=(12, 6))
    plt.bar(workflows, improvements)
    plt.title('各工作流行程时间改进情况')
    plt.xlabel('工作流类型')
    plt.ylabel('时间改进 (秒)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('result/workflow_improvements.png')
    plt.show()
```

### 🔄 结果应用

#### 实际部署建议
1. **逐步实施**: 先在部分科室试点优化方案
2. **效果监控**: 实施后监控实际行程时间变化
3. **反馈调整**: 根据实际使用情况调整工作流权重
4. **定期优化**: 定期重新运行优化以适应变化

#### 与现有系统集成
```python
def apply_optimized_layout(layout_file):
    """将优化结果应用到现有系统"""
    
    with open(layout_file, 'r') as f:
        optimized_layout = json.load(f)
    
    # 更新系统配置
    for function, spaces in optimized_layout['function_to_spaces'].items():
        update_function_assignment(function, spaces)
    
    # 验证更新结果
    validate_layout_consistency()
    
    print("优化布局已成功应用到系统中")
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
