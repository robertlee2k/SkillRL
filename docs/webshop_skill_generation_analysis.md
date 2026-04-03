# WebShop 技能生成系统详细解析

本文档详细解析 `skill_generation/webshop.py` 的逻辑结构、数据来源及与其他模块的关系。

---

## 1. 整体架构

这是一个从 WebShop agent 轨迹数据生成 Claude 风格技能的系统，使用 o3 API 作为生成引擎。

**核心流程**：
```
预存轨迹数据 → 分类提取 → LLM 抽象化 → 技能蒸馏 → claude_style_skills.json
```

---

## 2. 核心组件解析

### 2.1 OpenAIClient 类 (46-62行)

```python
class OpenAIClient:
    def __init__(self, max_new_tokens: int = 4096, model: str = "o3"):
        self.client = AzureOpenAI(...)
```

**作用**：封装 Azure OpenAI API 调用，专门使用 `o3` 模型

**关键点**：
- 使用 `max_completion_tokens` 而非 `max_tokens`（o3 模型特有）
- API 配置（密钥、endpoint）被硬编码为空字符串（实际使用时需填充）

---

### 2.2 数据加载与分类

#### load_memories (65-72行)

```python
def load_memories(json_paths: List[str]) -> List[Dict]:
```

**作用**：合并多个 JSON 文件的 memory 数据

**输入**：`generated_memories_webshop_100.json` + `generated_memories_webshop_101-200.json`
**输出**：扁平化的 memory 列表

#### categorize_by_product_type (75-160行)

```python
def categorize_by_product_type(memories: List[Dict]) -> Dict[str, Dict[str, List]]:
```

**作用**：按产品类型和结果分类

**分类逻辑**：
```
memory → 检查 goal 中的关键词 → 匹配产品类型 → 分到 success/failure
```

**7 种产品类型及关键词**：

| 类型 | 关键词示例 |
|------|-----------|
| apparel | shirt, dress, pants, jacket, sweater, coat, shorts, hoodie, blouse, jeans, t-shirt, suit |
| footwear | shoe, boot, sneaker, sandal, heel, slipper, loafer |
| home_decor | pillow, curtain, rug, blanket, lamp, vase, candle, towel, sheet, duvet |
| electronics | phone, laptop, tablet, headphone, speaker, camera, keyboard, monitor, charger |
| accessories | bag, wallet, watch, jewelry, belt, hat, scarf, glove, sunglasses, backpack |
| beauty_health | makeup, skincare, lotion, shampoo, conditioner, perfume, cream, serum, mask |
| other | 其他所有未匹配项 |

**匹配流程**：顺序检查每个关键词列表，首次匹配即停止

---

### 2.3 模式提取

#### extract_patterns (163-188行)

```python
def extract_patterns(memories: List[Dict], limit: int = 10) -> str:
```

**作用**：从 memory 提取精简模式供 o3 分析

**提取内容**：
```python
pattern = {
    'goal': original_goal,
    'steps': [{'action': ..., 'reasoning': ...}]  # 前5步
    'planning_pattern': planning_pattern,
    'mistakes': mistakes[:3]
}
```

**处理嵌套结构**：
```python
# 处理 strategic_guidelines 可能的双重嵌套
if isinstance(strategic, dict) and 'strategic_guidelines' in strategic:
    strategic = strategic['strategic_guidelines']
```

---

### 2.4 三类 Skills 生成

#### generate_general_skills (191-254行)

```python
def generate_general_skills(client: OpenAIClient, categorized_memories: Dict) -> List[Dict]:
```

**流程**：
```
1. 从各产品类型收集 success/failure 样本（各取前8条）
2. extract_patterns 精简为 JSON
3. 构建 prompt 发送 o3
4. 解析 JSON 响应
```

**Prompt 设计要点**：
- 强调 **Concise**（1-2句）、**Actionable**、**Transferable**
- 关注 6 大维度：搜索策略、产品选择、选项配置、约束验证、导航模式、价格处理

**Prompt 模板结构**：
```
You are an expert skill extraction system...

SUCCESSFUL TRAJECTORIES:
{success_patterns}

FAILED TRAJECTORIES:
{failure_patterns}

Generate 8-12 GENERAL SKILLS that apply across ALL product types...

Output as JSON list with format:
[
  {
    "skill_id": "gen_001",
    "title": "Short title (3-5 words)",
    "principle": "The core actionable insight in 1-2 sentences",
    "when_to_apply": "Specific trigger condition"
  }
]
```

#### generate_task_specific_skills (257-318行)

```python
def generate_task_specific_skills(client: OpenAIClient, product_type: str, ...) -> List[Dict]:
```

**差异点**：
- 为每个产品类型单独生成 4-6 个技能
- 包含产品类型描述上下文
- skill_id 使用产品类型前缀（如 `app_001`）

**产品类型描述映射**：
| 前缀 | 描述 |
|------|------|
| app | Apparel - clothing items requiring size, color, fit configuration |
| foot | Footwear - shoes requiring size and sometimes width/style options |
| home | Home Decor - decorative items often requiring style/color matching |
| elec | Electronics - tech products requiring specs, compatibility checks |
| acc | Accessories - fashion items requiring style, size, material checks |
| beauty | Beauty/Health - personal care items requiring ingredient, brand checks |
| other | General products |

#### generate_common_mistakes (321-385行)

```python
def generate_common_mistakes(client: OpenAIClient, categorized_memories: Dict) -> List[Dict]:
```

**特点**：
- 从成功案例的 `mistakes_to_avoid` 反推潜在错误
- 输出结构包含 `why_it_happens` 和 `how_to_avoid`

**输出结构**：
```json
{
  "mistake_id": "err_001",
  "description": "What the mistake is",
  "why_it_happens": "Root cause analysis",
  "how_to_avoid": "Prevention strategy"
}
```

---

### 2.5 主流程 main() (388-468行)

**执行顺序**：
```
┌─────────────────────────────────────────────────────────────┐
│  1. 加载 memories (2个JSON文件)                              │
│  2. 分类 by product_type + outcome                           │
│  3. 生成 general_skills (跨产品通用)                          │
│  4. 为每个 product_type 生成 task_specific_skills            │
│  5. 生成 common_mistakes                                     │
│  6. 保存到 claude_style_skills.json                          │
└─────────────────────────────────────────────────────────────┘
```

**命令行参数**：
```bash
python skill_generation/webshop.py \
    --memory_path memory_data/webshop/generated_memories_webshop_100.json \
    --output_path memory_data/webshop/claude_style_skills.json
```

---

### 2.6 JSON 解析策略 (245-254行)

所有生成函数使用统一的解析方式：
```python
json_start = response.find('[')
json_end = response.rfind(']') + 1
json.loads(response[json_start:json_end])
```

**原因**：o3 可能返回带额外文本的响应，需要定位纯 JSON 部分

---

## 3. 输出结构

### 3.1 claude_style_skills.json 结构

```json
{
  "general_skills": [
    {
      "skill_id": "gen_001",
      "title": "Constraint-Encoded Search",
      "principle": "Include all known constraints directly in the initial search query...",
      "when_to_apply": "When multiple constraints are specified in the goal"
    }
  ],
  "task_specific_skills": {
    "apparel": [
      {
        "skill_id": "app_001",
        "title": "Size-First Configuration",
        "principle": "...",
        "when_to_apply": "..."
      }
    ],
    "footwear": [...],
    "home_decor": [...],
    "electronics": [...],
    "accessories": [...],
    "beauty_health": [...],
    "other": [...]
  },
  "common_mistakes": [
    {
      "mistake_id": "err_001",
      "description": "...",
      "why_it_happens": "...",
      "how_to_avoid": "..."
    }
  ],
  "metadata": {
    "source": "generated from WebShop trajectories using o3",
    "total_memories_analyzed": 200,
    "product_distribution": {
      "apparel": 45,
      "footwear": 30,
      ...
    }
  }
}
```

### 3.2 Skill 结构字段说明

| 字段 | 说明 | 约束 |
|------|------|------|
| skill_id | 唯一标识 | `gen_XXX` / `app_XXX` / `dyn_XXX` |
| title | 简短标题 | 3-5 个词 |
| principle | 核心洞察 | 1-2 句可执行描述 |
| when_to_apply | 触发条件 | 明确的应用场景 |

---

## 4. 与 skill_updater.py 的关系

### 4.1 核心关系对比

这两个文件是 **互补的技能生成系统**，分别处理不同场景：

| 维度 | `webshop.py` | `skill_updater.py` |
|------|--------------|-------------------|
| **场景** | 离线预处理 | 在线运行时 |
| **触发时机** | 一次性脚本执行 | Agent 失败后动态触发 |
| **输入来源** | 预存的 JSON memory 文件 | 当前 session 的失败轨迹 |
| **生成目标** | 全量技能库初始化 | 增量补充缺失技能 |
| **LLM 后端** | Azure OpenAI `o3` | 火山引擎 Doubao |
| **ID 格式** | `gen_XXX` / `app_XXX` | `dyn_XXX` |

### 4.2 数据流向关系图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        webshop.py (离线初始化)                               │
│                                                                              │
│  memory_data/webshop/*.json  ──→  categorize  ──→  claude_style_skills.json │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓ 初始化技能库

┌─────────────────────────────────────────────────────────────────────────────┐
│                     skill_updater.py (在线增量)                              │
│                                                                              │
│  Agent 失败轨迹  ──→  analyze_failures()  ──→  dyn_001, dyn_002...          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓ 动态添加到技能库

            SkillsOnlyMemory.add_skills(new_skills)
```

### 4.3 协作模式总结

| 角色 | 文件 |
|------|------|
| **初始化器** | `webshop.py` — 从历史数据构建基础技能库 |
| **增量更新器** | `skill_updater.py` — 在线填补运行时发现的技能缺口 |
| **技能库载体** | `claude_style_skills.json` — 被 `SkillsOnlyMemory` 加载使用 |

两者形成 **"离线预训练 + 在线微调"** 的闭环模式。

---

## 5. 离线轨迹数据的来源

### 5.1 数据存储位置

```
memory_data/
├── webshop/
│   ├── generated_memories_webshop_100.json      (265KB, ~100条)
│   ├── generated_memories_webshop_101-200.json  (267KB, ~100条)
│   └── claude_style_skills.json                 (26KB, 技能库)
├── alfworld/
│   ├── generated_memories_alfworld_total.json
│   └── claude_style_skills.json
├── search/
│   ├── generated_memories_search.json
│   └── claude_style_skills.json
└── prompt/
    └── prompt.txt                               (LLM转换模板)
```

### 5.2 数据生成流程（三阶段）

根据 README.md (134行) 的描述：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Phase 1: Base Model 运行                                                    │
│                                                                              │
│  SFT Model (如 Jianwen/Webshop-7B-SFT) 在 WebShop 环境中执行任务              │
│  → 产生原始轨迹（action sequence + observation）                             │
│  → 记录 Success / Failure 结果                                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓

┌─────────────────────────────────────────────────────────────────────────────┐
│  Phase 2: LLM 抽象化处理                                                     │
│                                                                              │
│  使用 prompt.txt 中的模板，调用 LLM 将原始轨迹转换为结构化 memory：            │
│                                                                              │
│  • contextual_description  → 任务类型摘要                                    │
│  • refined_trajectory      → 反向因果链精简 + 泛化抽象                       │
│  • strategic_guidelines    → 成功模式 / 失误教训                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓

┌─────────────────────────────────────────────────────────────────────────────┐
│  Phase 3: 技能蒸馏                                                           │
│                                                                              │
│  skill_generation/webshop.py                                                 │
│  → 从 generated_memories 提取 patterns                                       │
│  → 调用 o3 API 生成 claude_style_skills.json                                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.3 prompt.txt 的核心作用

这是 **原始轨迹 → 结构化 memory** 的转换模板，包含 4 个关键 prompt：

| Prompt | 功能 |
|--------|------|
| `contextual_description` | 生成任务类型摘要（WebShop/ALFWorld/Search 分类） |
| `refined_trajectory` | **反向因果链算法**：从终点回溯，提取最小关键步骤，并**泛化抽象** |
| `semantic_knowledge` | 提取属性匹配规则 |
| `strategic_guidelines` | 成功时提取 `planning_pattern`，失败时提取 `mistakes_to_avoid` |

**泛化抽象示例**（prompt.txt 149行）：
```
原始值: "$50.00", "Red", "5x-large"
抽象后: [Price_Constraint], [Color_Constraint], [Size_Constraint]
```

### 5.4 Memory 数据结构示例

```json
{
  "memory_id": "mem_webshop_19499503",
  "contextual_description": "WebShop task to purchase a Men's Apparel item with Color, Size, Material, Fit, Sleeve Style, and Price constraints. Solved by searching with detailed terms, selecting size and color options, and buying.",
  "tags": {
    "environment": "Webshop",
    "outcome": "Success"
  },
  "content": {
    "task_meta": {
      "original_goal": "Find me machine wash men's dress shirts with cotton spandex, classic fit, short sleeve with color: melon berry, and size: large, and price lower than 50.00 dollars."
    },
    "refined_trajectory": {
      "refined_trajectory": [
        {
          "step_index": 0,
          "action": "search[men's dress shirts cotton spandex classic fit short sleeves [Color_Constraint] [Size_Constraint] [Price_Constraint] or less]",
          "critical_observation": "Search results page shows multiple apparel items...",
          "reasoning": "Formulate a search query that encodes all known attribute constraints..."
        },
        {
          "step_index": 1,
          "action": "click[apparel_item]",
          "critical_observation": "Product detail page for a men's short-sleeve shirt...",
          "reasoning": "Open a promising apparel item..."
        }
      ]
    },
    "strategic_guidelines": {
      "strategic_guidelines": {
        "planning_pattern": "search -> click_product -> set_size -> set_color -> purchase",
        "mistakes_to_avoid": []
      }
    }
  },
  "origin_env_id": "env001"
}
```

### 5.5 原始轨迹数据说明

仓库中 **只保留了 processed memories**，原始轨迹数据不在仓库中。

**推测原因**：
- 原始轨迹体积大、噪声多，保存价值低
- `refined_trajectory` 已通过反向因果链精简，保留核心信息
- `origin_env_id` 字段（如 `"env001"`）暗示原始文件存在但未公开

---

## 6. 关键设计思想总结

| 设计点 | 说明 |
|--------|------|
| **分层设计** | 通用技能 + 任务特定技能，覆盖不同粒度 |
| **失败驱动** | 从 failure 轨迹提炼避免策略，转化为负面约束 |
| **简洁原则** | 强制 o3 生成 1-2 句可执行规则，避免冗长描述 |
| **产品领域化** | 关键词匹配实现自动化分类，领域适配 |
| **泛化抽象** | 具体值替换为约束类型，提升跨任务迁移能力 |
| **反向因果链** | 从终点回溯提取关键步骤，剔除噪声动作 |

---

## 7. 相关文件索引

| 文件 | 路径 | 作用 |
|------|------|------|
| 技能生成脚本 | `skill_generation/webshop.py` | 离线批量生成技能 |
| 技能更新脚本 | `agent_system/memory/skill_updater.py` | 在线增量更新技能 |
| 技能库载体 | `memory_data/webshop/claude_style_skills.json` | 最终技能存储 |
| Memory 数据 | `memory_data/webshop/generated_memories_*.json` | 预处理轨迹数据 |
| LLM 模板 | `memory_data/prompt/prompt.txt` | 轨迹抽象化 prompt |
| 技能内存类 | `agent_system/memory/skills_only_memory.py` | 运行时加载技能库 |

---

## 8. 使用示例

### 8.1 生成技能库

```bash
python skill_generation/webshop.py \
    --memory_path memory_data/webshop/generated_memories_webshop_100.json \
    --output_path memory_data/webshop/claude_style_skills.json
```

### 8.2 RL 训练中使用技能库

```bash
export MODEL_PATH=YOUR_SFT_CKPT
bash examples/grpo_trainer/run_webshop_skills.sh
```

关键配置参数：
```
+env.use_skills_only_memory=True
+env.skills_only_memory.skills_json_path=memory_data/webshop/claude_style_skills.json
+env.skills_only_memory.top_k=6
+env.skills_only_memory.enable_dynamic_update=True
+env.skills_only_memory.update_threshold=0.4
+env.skills_only_memory.max_new_skills=3
```

---

*文档生成时间：2026-03-29*
*基于 SkillRL 仓库代码分析*