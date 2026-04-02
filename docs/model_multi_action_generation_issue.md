# 模型多动作生成问题分析

## 问题现象

在沙盒推演中，模型一次推理输出了多个 `<action>` 标签，导致解析失败或选择错误的动作。

## 真实数据示例

### 1. 模型原始输出（问题示例）

**输入场景：** 售后场景，买家说"车轮断了，我要退货"

**模型原始输出（未截断）：**

```
<action>aft_initiate_exchange</action> 根据买家的描述，我们需要进一步了解具体情况并发起换货流程。请稍等，我将为您处理。
<action>aft_initiate_exchange</action> 我会帮您发起退货申请，请提供您的订单号以便我们更快地处理。 
<action>gen_transfer</action> 如果您有任何其他问题，或者需要进一步的帮助，我会尽快转接给您的人工客服。请提供您的订单号。 
<action>aft_initiate_exchange</action> 请提供您的订单号，以便我们能够更快地处理您的退货请求。 
... (循环重复，直到达到 max_new_tokens 限制)
```

**统计信息：**
- `<action>` 标签数量: **18个**
- 思考标签数量: 1个（在输出开头）
- 输出 token 数量: 512（达到最大限制）
- EOS token ID: 151645（模型没有主动停止）

### 2. 训练数据 Prompt 格式

**System 消息关键部分：**

```
【动作英文名称】: gen_clarify
  功能说明: 澄清意图，确认买家需求
  ...

<action>在此填写最终选择的动作英文名称</action>

【⚠️ 格式警告 ⚠️】你的动作必须、且只能从下方的技能列表中，准确复制【动作英文名称】后面的纯英文拼写。
```

**User 消息：**

```
## 当前对话状态

**场景类型**: aftersale
**买家情绪**: neutral

**买家最新消息**:
订单号:1914415214273215577 共1件商品,合计￥656.49 交易时间:2023-06-15 23:58:10

**当前槽位状态**:
- order_id_collected: True

## 任务
请分析买家的需求，并结合系统提供的可用动作列表与约束规则，选择最合适的唯一客服【动作英文名称】。
```

### 3. 训练数据中的失败样本分析

检查训练 bad cases：

| Step | Bad Cases 数量 | 无 action 标签 | 单 action 标签 | 多 action 标签 |
|------|---------------|---------------|---------------|---------------|
| 60   | 532           | 532           | 0             | 0             |
| 100  | 426           | 426           | 0             | 0             |
| 150  | 307           | 307           | 0             | 0             |

**发现：** 所有 bad cases 都是因为**没有输出 action 标签**，而不是输出多个 action。

### 4. 训练数据结构

训练数据只包含 prompt，没有 response（RL训练模式）：

```
Columns: ['data_source', 'prompt', 'ability', 'env_kwargs', 'extra_info']
Shape: (4608, 5)
```

prompt 是一个 chat 格式的列表，包含 system 和 user 两条消息。

## 问题分析

### 模型行为模式

从真实输出可以看到，模型的生成模式是：

1. 输出思考过程
2. 输出第一个 `<action>xxx</action>`
3. 输出动作对应的自然语言回复
4. **继续生成**第二个 `<action>xxx</action>`
5. **继续生成**第三个 `<action>xxx</action>`
6. 循环直到达到 token 限制

### 根本原因分析

#### 1. Rollout 生成没有配置停止条件

检查 `verl/workers/rollout/vllm_rollout/vllm_rollout.py`:

```python
kwargs = dict(
    n=1,
    logprobs=0,
    max_tokens=config.response_length,
)
self.sampling_params = SamplingParams(**kwargs)
```

vllm 的 `SamplingParams` 支持 `stop` 参数：

```python
SamplingParams(
    stop=[],           # 停止字符串列表
    stop_token_ids=[], # 停止 token ID 列表
    ...
)
```

**但当前配置没有使用这些参数！**

#### 2. Projection 只提取第一个 action

`customer_service_projection` 函数使用正则表达式 `search()` 只提取第一个 `<action>` 标签：

```python
match = re_action_block.search(action)  # 只匹配第一个
```

这意味着：
- 如果第一个动作是有效的，模型得到正奖励
- 后续的多余输出没有被惩罚

#### 3. 环境只处理一个动作

`CustomerServiceEnv.step()` 只接收一个动作，处理后就返回奖励。模型不知道它输出了多个动作。

### 问题链条

```
模型生成多个 <action> 
    -> Projection 只提取第一个
    -> 环境只处理第一个
    -> 模型得到奖励，没有惩罚
    -> 模型学会了"继续生成没关系"
    -> 循环重复
```

## 解决方案

### 方案1：在 Rollout 中添加停止条件（推荐）

修改 `verl/workers/rollout/vllm_rollout/vllm_rollout.py`:

```python
kwargs = dict(
    n=1,
    logprobs=0,
    max_tokens=config.response_length,
    stop=["</action>"],  # 添加停止条件
    include_stop_str_in_output=True,  # 保留停止字符串
)
```

或者在配置文件中添加：

```yaml
actor_rollout_ref:
  rollout:
    name: vllm
    # ... 其他配置
    stop:
      - "</action>"
    include_stop_str_in_output: true
```

### 方案2：修改 Reward 函数

在 `customer_service_projection` 或 `customer_service_fallback_projection` 中：

```python
# 检查是否有多个 <action> 标签
action_count = len(re.findall(r'<action>', action))
if action_count > 1:
    # 惩罚多动作输出
    valids[i] = 0
    logger.warning(f"[projection] Multiple actions detected: {action_count}")
```

### 方案3：修改环境惩罚

在 `CustomerServiceEnv.step()` 中，检查原始模型输出：

```python
# 在 step 函数开始时
raw_action_count = action.count('<action>')
if raw_action_count > 1:
    # 额外惩罚
    step_reward -= 0.5 * (raw_action_count - 1)
```

## 验证方法

### 1. 测试停止条件

```python
from vllm import SamplingParams

params = SamplingParams(
    max_tokens=512,
    stop=["</action>"],
    include_stop_str_in_output=True,
)
# 运行推理，检查是否在第一个 </action> 后停止
```

### 2. 检查训练日志

```bash
# 查看 bad cases 中多动作的比例
python3 -c "
import json
with open('outputs/bad_cases/bad_cases_step150_20260402_070717.json') as f:
    data = json.load(f)
print(f'Total bad cases: {len(data["bad_cases"])}')
"
```

## 临时解决方案（已实施）

在推理时截断：检测到 `</action>` 后立即停止生成。

```python
# viewer/services/sandbox_service.py
if '</action>' in response:
    end_pos = response.find('</action>') + len('</action>')
    response = response[:end_pos].rstrip()
```
