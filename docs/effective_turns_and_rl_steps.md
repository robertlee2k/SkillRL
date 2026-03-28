# 有效对话轮次与 RL Steps 技术文档

## 概述

本文档说明 `effective_turn_count` 和 `rl_steps` 两个关键字段的定义、计算逻辑及其在强化学习训练中的应用。

## 字段定义

### effective_turn_count

**定义**：经过数据清洗后的对话消息总数（User + Agent turns 合计）

**含义**：
- 原始对话数据中，同一角色可能连续发送多条消息
- 经过 `etl/aggregator.py` 的聚合逻辑后，连续同角色消息被合并
- 最终形成严格的 User-Agent 交替结构

### rl_steps

**定义**：Agent 需要执行的 action 数量，即 User turns 的数量

**含义**：
- 在 RL 环境中，每个 User turn 触发一次 Agent 响应
- 每个 Agent 响应对应一个 action（skill 选择）
- `rl_steps` 直接对应 `max_steps` 配置的限制

## 关系公式

```
rl_steps = count(User turns) = floor((effective_turn_count + 1) / 2)
```

对于标准 User-Agent 交替对话：
- `rl_steps ≈ effective_turn_count / 2`
- 以 User 开头的对话：`rl_steps = ceil(effective_turn_count / 2)`

## 计算逻辑

### 数据清洗流程

```
原始消息列表
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 1: aggregate_turns()                                   │
│  - 过滤 MARKETING 消息                                       │
│  - 处理 SYSTEM 消息（提取槽位）                               │
│  - 合并连续同角色消息                                         │
│  - QA/ASSISTANT → Agent, BUYER → User                       │
│  - 输出：严格 User-Agent 交替的 turns 列表                    │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 2: 处理 Agent-first Session                            │
│  - 如果 turns 以 Agent 开头，删除开头的 Agent turns          │
│  - 直到找到第一个 User turn                                   │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 3: 计算                                                │
│  - effective_turn_count = len(turns)                        │
│  - rl_steps = count(turns where role == 'User')             │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
最终 Playbook 结构
```

### 核心代码

```python
# etl/aggregator.py - 消息聚合逻辑
def aggregate_turns(messages):
    turns = []
    current_role = None
    current_texts = []

    for msg in messages:
        # 跳过 MARKETING
        if msg.get('sent_by') == 'MARKETING':
            continue

        # 角色映射
        mapped_role = ROLE_MAPPING.get(msg.get('sent_by'))

        # 同角色连续，拼接文本
        if current_role == mapped_role:
            current_texts.append(text)
        else:
            # 角色切换，保存上一轮
            if current_role:
                turns.append({
                    'role': current_role,
                    'text': ' '.join(current_texts)
                })
            current_role = mapped_role
            current_texts = [text]

    # 保存最后一轮
    if current_role:
        turns.append({'role': current_role, 'text': ' '.join(current_texts)})

    return {'turns': turns}
```

```python
# etl/pipeline.py - 字段计算
def build_playbook(session):
    turns = session.get('turns', [])

    # 总 turns 数
    effective_turn_count = len(turns)

    # RL steps = User turns 数量
    user_turn_count = sum(1 for t in turns if t['role'] == 'User')
    rl_steps = user_turn_count

    playbook = {
        'effective_turn_count': effective_turn_count,
        'rl_steps': rl_steps,
        ...
    }
```

## 示例说明

### 示例 1：简单对话

原始消息：
```
1. [BUYER] 你好，这款产品多少钱？
2. [QA] 亲，这款是 99 元哦
3. [BUYER] 好的，我要买
4. [ASSISTANT] 好的，马上为您安排
```

处理后 turns：
```
1. [User] 你好，这款产品多少钱？
2. [Agent] 亲，这款是 99 元哦
3. [User] 好的，我要买
4. [Agent] 好的，马上为您安排
```

统计：
- `effective_turn_count = 4`
- `rl_steps = 2`（User turns: 第1、3条）

### 示例 2：连续同角色消息合并

原始消息：
```
1. [BUYER] 在吗？
2. [BUYER] 想问下发货时间
3. [ASSISTANT] 在的呢亲
4. [QA] 一般 24 小时内发货
5. [BUYER] 好的谢谢
6. [BUYER] 还有一个问题
```

处理后 turns（连续消息合并）：
```
1. [User] 在吗？ 想问下发货时间
2. [Agent] 在的呢亲 一般 24 小时内发货
3. [User] 好的谢谢 还有一个问题
```

统计：
- `effective_turn_count = 3`
- `rl_steps = 2`（User turns: 第1、3条）

### 示例 3：Agent-first Session 处理

原始消息：
```
1. [ASSISTANT] 欢迎光临本店！
2. [QA] 有什么可以帮您的吗？
3. [BUYER] 我想买个奶瓶
4. [ASSISTANT] 好的，请问宝宝多大？
5. [BUYER] 6个月
```

处理后 turns（删除开头的 Agent turns）：
```
1. [User] 我想买个奶瓶
2. [Agent] 好的，请问宝宝多大？
3. [User] 6个月
```

统计：
- `effective_turn_count = 3`
- `rl_steps = 2`

## RL 训练应用

### max_steps 配置

在 veRL 训练配置中：
```bash
env.max_steps=20
```

这表示 Agent 最多执行 20 个 action，即最多处理 20 个 User turns。

### 数据过滤

为确保对话不会被截断，应在数据准备阶段过滤超长对话：

```python
# 过滤会被截断的对话
valid_playbooks = [
    pb for pb in playbooks
    if pb['rl_steps'] <= max_steps
]
```

### 当前数据统计

基于 `outputs/playbooks_all.json`（5,784 条）：

| 指标 | effective_turn_count | rl_steps |
|------|---------------------|----------|
| 最小值 | 2 | 1 |
| 最大值 | 42 | 21 |
| 平均值 | 9.3 | 4.7 |
| P90 | 22 | 11 |
| P95 | 26 | 13 |
| P99 | 31 | 16 |

| 阈值 | 超标数量 | 占比 |
|------|---------|------|
| rl_steps > 20（会被截断） | 1 | 0.02% |
| rl_steps > 15 | 71 | 1.2% |
| rl_steps > 10 | 618 | 10.7% |

**结论**：对于 `max_steps=20`，仅 1 条 session 会被截断，保留率 99.98%。

## 相关文件

| 文件 | 作用 |
|------|------|
| `etl/aggregator.py` | 消息聚合逻辑，合并连续同角色消息 |
| `etl/cleaner.py` | Session 清洗，处理 Agent-first 情况 |
| `etl/pipeline.py` | Playbook 生成主流程，计算并写入字段 |
| `scripts/backfill_turn_counts.py` | 存量数据回填脚本 |
| `scripts/verify_turn_counts.py` | 验证脚本，校验回填正确性 |

## 更新历史

| 日期 | 更新内容 |
|------|---------|
| 2026-03-28 | 初始版本，定义 `effective_turn_count` 和 `rl_steps` 字段 |