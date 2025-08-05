# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 语言设置

**重要：Claude Code 在此项目中必须使用中文回复。**

## 项目概述

医院房间检测与布局优化系统 - 基于图像处理和强化学习的多层医院网络生成与布局优化工具。

### 核心架构

这是一个双阶段优化系统：

1. **网络生成阶段** (`main.py`)：
   - 从楼层平面图生成多层医院网络图
   - 图像处理：识别房间、走廊、门等区域
   - 网络构建：创建节点和边的图结构
   - 行程时间计算：生成房间间行程时间矩阵

2. **强化学习优化阶段** (`rl_main.py`)：
   - 基于PPO算法的布局优化
   - 考虑就医流程和相邻性约束
   - 动态调整科室布局以优化总体效率

### 关键模块

- `src/network/`: 网络生成核心模块
  - `network.py`: 单层网络生成器
  - `super_network.py`: 多层网络管理器
  - `node_creators.py`: 各类节点创建策略
- `src/rl_optimizer/`: 强化学习优化器
  - `agent/ppo_agent.py`: PPO智能体实现（支持动态学习率调度）
  - `env/layout_env.py`: 布局优化环境
  - `data/cache_manager.py`: 数据缓存管理
  - `utils/lr_scheduler.py`: 学习率调度器工具
- `src/analysis/`: 分析工具
  - `travel_time.py`: 行程时间计算
  - `process_flow.py`: 就医流程分析
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

### 网络生成
```bash
# 生成多层医院网络（主要功能）
uv run python main.py

# 输出文件：
# - result/super_network_3d.html (可视化)
# - result/super_network_travel_times.csv (行程时间矩阵)
```

### 强化学习优化
```bash
# 训练布局优化模型
uv run python rl_main.py --mode train

# 使用已训练模型进行评估
uv run python rl_main.py --mode evaluate --model-path logs/ppo_layout_YYYYMMDD-HHMMSS/final_model.zip

# 训练输出：
# - logs/ppo_layout_*/: 训练日志和模型
# - logs/ppo_layout_*/tensorboard_logs/: TensorBoard日志
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

### RLConfig (src/config.py:136-233)
- 强化学习超参数
- 训练配置
- 约束设置
- **学习率调度器配置**: 支持线性衰减和常数学习率

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

- 2025-08-05: 初始创建动态CLAUDE.md文档，包含完整架构概述和uv命令格式
- 2025-08-05: 添加重要开发注意事项，包含颜色映射系统、数据预处理要求和性能考虑
- 2025-08-05: 完善技术栈信息，添加MaskablePPO等具体实现细节
- 2025-08-05: **引入线性衰减学习率调度器** - 添加lr_scheduler.py工具模块，支持线性衰减和常数学习率调度，提高训练稳定性和收敛性
- 2025-08-05: 添加配置管理规范：所有硬编码的配置项均应放置在config.py文件中