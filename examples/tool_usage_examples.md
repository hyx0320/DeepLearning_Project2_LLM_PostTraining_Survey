# Tool Usage Examples：大模型微调、对齐与评估工具实践

> 本文档整合了《Fine-tuning 技术笔记》《Alignment 技术笔记》和《Evaluation Harness 技术深度解析》三份笔记的核心方法，以 **"解决什么问题 → 用什么数据 → 优化目标是什么 → 优点与局限"** 的框架组织，并附上可运行的工具代码示例。
>
> 不要求完整推导所有数学公式，重在讲清每种方法的定位、数据需求和核心权衡。

---

## 目录

1. [概述：三种工具的定位与关系](#1-概述三种工具的定位与关系)
2. [Fine-tuning 工具](#2-fine-tuning-工具)
   - [2.1 Full Fine-tuning（全量微调）](#21-full-fine-tuning全量微调)
   - [2.2 LoRA（低秩适配）](#22-lora低秩适配)
   - [2.3 QLoRA（量化低秩适配）](#23-qlora量化低秩适配)
   - [2.4 LoRA 改进变体：DoRA / LoRA+ / PiSSA](#24-lora-改进变体)
3. [Alignment 工具](#3-alignment-工具)
   - [3.1 RLHF（PPO）](#31-rlhf-ppo)
   - [3.2 DPO（直接偏好优化）](#32-dpo直接偏好优化)
   - [3.3 DPO 改进变体：ORPO / SimPO](#33-dpo-改进变体)
   - [3.4 GRPO（组相对策略优化）](#34-grpo组相对策略优化)
   - [3.5 Constitutional AI（宪法 AI） / RLAIF](#35-constitutional-ai--rlaif)
4. [Evaluation 工具](#4-evaluation-工具)
   - [4.1 EleutherAI lm-evaluation-harness](#41-eleuther-ai-lm-evaluation-harness)
   - [4.2 LLM-as-Judge 评估模式](#42-llm-as-judge-评估模式)
5. [综合管线示例](#5-综合管线示例)

---

## 1. 概述：三种工具的定位与关系

在构建一个大语言模型应用时，通常需要经过以下三个阶段，每个阶段都有对应的工具和方法：

```
预训练基座模型 (Base Model)
       │
       ▼
┌───────────────────┐
│  Fine-tuning 阶段   │  ← 教会模型"回答问题"或适应特定领域
│  (SFT / LoRA / QLoRA) │     数据：指令-回复对
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Alignment 阶段     │  ← 让模型输出更符合人类偏好
│  (DPO / RLHF / GRPO)│     数据：偏好对 (chosen vs rejected)
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Evaluation 阶段    │  ← 系统评估模型在各维度的表现
│  (lm-eval harness) │     数据：标准化 Benchmark
└───────────────────┘
```

这三类工具不是孤立的——**微调决定能力基础，对齐决定行为偏好，评估决定迭代方向**。实际项目通常在这三者间循环迭代。

> 🔗 **关联笔记**：
> - 微调方法的理论基础、数据格式、超参数调优 → 详见《Fine-tuning 技术笔记》
> - 对齐方法的数学推导、方法对比、选型指南 → 详见《Alignment 技术笔记》
> - 评估框架的架构、Benchmark 详解、结果解读 → 详见《Evaluation Harness 技术深度解析》

---

## 2. Fine-tuning 工具

### 2.1 Full Fine-tuning（全量微调）

#### 解决什么问题

让基座模型适应**特定任务**或**特定领域**。基座模型只学会了"续写文本"，Full FT 让它学会"回答具体问题"或"完成特定格式的任务"。

#### 用什么数据

| 场景 | 数据示例 | 数据量推荐 |
|------|---------|-----------|
| 指令微调（SFT） | 指令-回复对（Alpaca / ChatML 格式） | 5K ~ 52K |
| 领域知识注入 | 领域问答对（如医疗 QA、法律文书） | 2K ~ 10K |
| 分类任务 | 文本-标签对 | 500+ / 类 |

**数据格式示例（ChatML）**：
```json
{
  "messages": [
    {"role": "user", "content": "什么是快速排序？"},
    {"role": "assistant", "content": "快速排序是一种分治排序算法..."}
  ]
}
```

#### 优化目标

使用标准的交叉熵损失（仅计算 assistant 回复部分）：

$$\mathcal{L}_{SFT} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T_i} \log P_\theta(y_{i,t} | x_i, y_{i,<t})$$

其中 $\theta$ 初始化为预训练权重，所有参数在训练中更新。

#### 优点

- 参数空间不受限制，理论上适应能力最强
- 效果上限最高（数据充足时）
- 实现简单，不引入额外模块

#### 局限

- **显存极高**：7B 模型需 ~112GB 显存（含激活值）——公式：$\text{参数} \times \text{16 bytes}$
- **存储成本高**：每任务一个完整模型（~14GB+）
- **灾难性遗忘风险高**：所有参数被修改，可能忘记预训练学到的通用知识
- **过拟合风险**：数据不足时效果反而不如 PEFT 方法

#### 代码示例（HuggingFace Trainer）

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

training_args = TrainingArguments(
    output_dir="./full_ft_output",
    num_train_epochs=3,
    per_device_train_batch_size=1,        # 全量微调 batch 必须小
    gradient_accumulation_steps=8,         # 有效 batch size = 8
    learning_rate=2e-5,                    # 全量微调用小学习率
    bf16=True,
    save_strategy="epoch",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=load_dataset("json", data_files="sft_data.jsonl")["train"],
    tokenizer=tokenizer,
)
trainer.train()
```

> ⚠️ **何时选择 Full FT？** 数据量 > 10K、有充足 GPU 资源（多卡 A100）、且需要最大化效果时。否则优先考虑 PEFT 方法。

---

### 2.2 LoRA（低秩适配）

#### 解决什么问题

Full FT 的显存和存储开销对个人开发者和中小企业不友好。LoRA 的核心洞察是：**微调中的权重更新 $\Delta W$ 具有低秩性质**（本征维度假说），因此可以用两个小矩阵 $BA$ 来参数化权重更新，从而大幅减少可训练参数量。

#### 用什么数据

与 Full FT 相同的数据格式，但数据量较少（500 ~ 10K 条）时 LoRA 比 Full FT 更鲁棒。

#### 优化目标

修改前向传播：$h = W_0 x + \frac{\alpha}{r} B A x$

- $W_0 \in \mathbb{R}^{d \times k}$：预训练权重（**冻结**）
- $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$：可训练的低秩矩阵
- $r \ll \min(d, k)$：典型值 8-16
- $\alpha$：缩放因子

损失函数与 SFT 相同（交叉熵），但仅优化 $A$ 和 $B$（通常占全部参数的 0.1%~1%）。

#### 优点

- **显存大幅降低**：7B 模型仅需 ~16GB（不含激活值），是 Full FT 的 1/7
- **存储极轻量**：每任务仅保存几 MB 的 LoRA 权重，无需保存完整模型
- **推理零额外开销**：LoRA 权重可合并回 $W_0$，计算量不变
- **可热插拔**：多任务时只需切换 LoRA 权重而非整个模型
- **隐式正则化**：低秩约束降低过拟合风险

#### 局限

- 对复杂任务（如数学推理）效果上限略低于 Full FT（差距通常 < 3%）
- 需要选择 `target_modules`（哪些矩阵用 LoRA）和秩 $r$，有一定调参成本
- 训练时仍需加载完整基座模型（虽然冻结）

#### 代码示例（PEFT 库）

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# LoRA 配置
lora_config = LoraConfig(
    r=8,                                           # 秩
    lora_alpha=32,                                 # 缩放因子 (α/r = 4)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 输出: trainable params: 8.4M || all params: 6.7B || trainable%: 0.12%

# 保存仅 ~16MB 的 LoRA 权重
model.save_pretrained("./my_lora_adapter")
```

> 💡 **经验法则**：数据量 < 1K 时用 $r=4$（更强的正则化）；数据量 > 10K 时用 $r=16$（更多表达能力）。

---

### 2.3 QLoRA（量化低秩适配）

#### 解决什么问题

LoRA 虽然减少了可训练参数，但基座模型仍需以 FP16 加载。对于 65B+ 模型，单卡仍无法承载。QLoRA 的目标是**在消费级硬件（RTX 3090/4090）上微调大模型**。

#### 核心技术（三大组件）

| 技术 | 作用 | 节省效果 |
|------|------|---------|
| **NF4 量化** | 将基座模型以 4-bit NormalFloat 精度加载 | 从 2 bytes/参数 → 0.5 bytes/参数 |
| **双重量化** | 对量化常数再做 8-bit 量化 | 额外节省 ~0.5 bit/参数 |
| **分页优化器** | GPU 显存不足时将优化器状态换出到 CPU | 防止 OOM |

#### 前向传播

$$h = \text{dequant}(W_0^{NF4}) \cdot x^{BF16} + \frac{\alpha}{r} (BA)^{BF16} \cdot x^{BF16}$$

基座模型存储为 NF4（反量化到 BF16 后计算），LoRA 参数以 BF16 精度存储和训练。

#### 优点

- **极低显存**：65B 模型仅需 ~33GB 显存（单卡 RTX 3090 可运行）
- **效果接近 LoRA**：原始论文显示 QLoRA 与同配置 LoRA 的效果差距 < 1%
- **大幅降低微调门槛**：个人开发者可在消费级硬件上处理百亿参数模型

#### 局限

- 训练速度比 LoRA 慢 20-30%（反量化开销）
- 对长序列训练不友好（分页优化器的换入换出增加延迟）
- 如果再做 GPTQ 等后训练量化，会引入双重量化误差累积

#### 代码示例（BitsAndBytes + PEFT）

```python
from transformers import BitsAndBytesConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# 加载 4-bit 量化模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)

# 配置 LoRA（QLoRA 通常用更大的秩）
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
)
model = get_peft_model(model, lora_config)

# 使用 SFTTrainer 进行训练
trainer = SFTTrainer(
    model=model,
    args=TrainingArguments(
        output_dir="./qlora_output",
        per_device_train_batch_size=4,
        learning_rate=2e-4,         # QLoRA 可用更高学习率
        bf16=True,
        logging_steps=10,
    ),
    train_dataset=dataset,
    tokenizer=tokenizer,
)
trainer.train()
```

#### 显存对比速查（7B 模型）

| 方法 | 显存需求 | 可运行硬件 |
|------|---------|-----------|
| Full FT (混合精度) | ~112 GB | 2×A100 80GB |
| LoRA (FP16) | ~16 GB | 1×A100 40GB / RTX 4090 |
| QLoRA (NF4) | ~6 GB | 1×RTX 3090 / RTX 4090 |

---

### 2.4 LoRA 改进变体

以下是在标准 LoRA 基础上涌现的改进方法，每种都针对 LoRA 的某个弱点做了专项优化。

| 方法 | 解决什么问题 | 核心改进 | 效果 vs LoRA | 推荐场景 |
|------|------------|---------|-------------|---------|
| **DoRA** | 幅度与方向更新耦合，限制表达能力 | 将权重分解为幅度 $m$ 和方向 $\Delta V$，幅度直接学习，方向用 LoRA | ↑ 1-3% | 推理/数学任务 |
| **LoRA+** | $A$ 和 $B$ 梯度方差差异大，统一 LR 效率低 | $A$ 和 $B$ 使用不同学习率（$\eta_B = 2\eta_A$） | → 收敛快 2× | 任何 LoRA 场景（即插即用） |
| **PiSSA** | 随机初始化 $A, B$ 导致收敛慢 | 对 $W_0$ 做 SVD 分解，用主成分初始化 $A, B$ | ↑ 1-3% | 少数据场景 |
| **AdaLoRA** | 所有层用相同 $r$，不高效 | 根据重要性分数动态分配秩 | ↑ 1-2% | 模块重要度差异大的模型 |
| **VeRA** | LoRA 参数量仍不够轻 | 共享随机矩阵 + 仅学缩放向量 | ↓ 2-5%（参数量 ↓90%） | 极端资源受限 |

#### DoRA 代码示例

```python
from peft import LoraConfig, get_peft_model

# DoRA 只需在 LoraConfig 中设置 use_dora=True
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    use_dora=True,      # 启用 DoRA
)
model = get_peft_model(model, lora_config)
```

---

## 3. Alignment 工具

### 3.1 RLHF（PPO）

#### 解决什么问题

SFT 教会模型"回答问题"，但无法区分"好的回答"和"差的回答"——因为 SFT 在损失函数中对所有非参考答案一视同仁。RLHF 的目标是让模型输出**符合人类偏好**：有用、无害、诚实。

#### 用三阶段流水线解决

```
阶段1: SFT               → 模型学会"回答"
阶段2: 训练 Reward Model  → 训练一个"裁判"来给回复打分
阶段3: PPO 强化学习       → 用裁判分数优化策略，同时约束不要偏离 SFT 太远
```

#### 用什么数据

| 阶段 | 数据 | 示例 |
|------|------|------|
| SFT（阶段1） | 高质量指令-回复对 | 同 Fine-tuning 的 SFT 数据 |
| RM 训练（阶段2） | 人类偏好排序数据 | 对同一 prompt 的 4-9 个回复，人类标注偏好顺序 |
| PPO（阶段3） | 新的 prompt 集合 | 不需要人工标注，模型生成 + RM 打分 |

**偏好数据格式**（RM 训练用）：
```json
{
  "prompt": "请解释什么是黑洞",
  "chosen": "黑洞是一种引力极强的天体...",    // 人类偏好
  "rejected": "黑洞就是一个黑色的洞..."        // 人类不偏好
}
```

#### 优化目标

**RM 训练**（Bradley-Terry 模型）：

$$\mathcal{L}_{RM}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}_{pref}} \left[ \log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l)) \right]$$

最大化 winner 与 loser 的分数差距。

**PPO 优化**（两项目标）：

$$\max_{\theta} \ \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)} \left[ r_\phi(x, y) - \beta \cdot D_{KL}\left(\pi_\theta(\cdot|x) \ \|\ \pi_{SFT}(\cdot|x)\right) \right]$$

- 第1项：最大化 RM 评分（让回复更符合人类偏好）
- 第2项：KL 惩罚（防止偏离 SFT 太远，避免 reward hacking）

#### 优点

- **经典且经过大规模验证**：ChatGPT 的方法论基础
- **在线探索**：PPO 在训练中不断生成新回复并得到实时反馈，可能发现超出数据覆盖范围的更好策略
- **灵活**：针对任意可量化的奖励进行优化

#### 局限

- **实现极其复杂**：需同时维护 4 个模型（Actor / Critic / RM / Reference），显存消耗极高
- **训练不稳定**：PPO 的超参数（$\beta$, clip $\epsilon$, KL penalty）敏感
- **Reward Hacking 风险**：模型可能学会"钻空子"拿高分（如生成啰嗦的废话来刷长度分）
- **分布偏移**：PPO 优化后模型的输出分布会改变，RM 在"陌生领域"打分不准确

#### 代码示例（TRL 库）

```python
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# 加载带有 Value Head 的 Actor 模型
model = AutoModelForCausalLMWithValueHead.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
ref_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
reward_model = AutoModelForSequenceClassification.from_pretrained("./my_rm")

# PPO 训练循环
ppo_config = PPOConfig(batch_size=16, learning_rate=1.41e-5, ppo_epochs=4, init_kl_coef=0.05)
ppo_trainer = PPOTrainer(config=ppo_config, model=model, ref_model=ref_model, dataset=dataset)

for batch in ppo_trainer.dataloader:
    queries = batch["query"]
    responses = ppo_trainer.generate(queries, top_p=0.9, temperature=0.7)
    rewards = reward_model(queries, responses)        # RM 打分
    stats = ppo_trainer.step(queries, responses, rewards)  # PPO 更新
```

---

### 3.2 DPO（直接偏好优化）

#### 解决什么问题

RLHF 太复杂了——需要训练 RM、需要 PPO、需要同时维护 4 个模型。DPO 的核心洞察是：**RLHF 的最优策略可以直接用一个分类损失来表示，不需要显式训练 RM，也不需要 RL**。

数学关键：配分函数 $Z(x)$ 在 Bradley-Terry 模型的减法中精确抵消了，使得 DPO 能够直接基于偏好对优化。

#### 用什么数据

与 RLHF 的 RM 训练阶段相同：偏好对数据（chosen vs rejected）。

```json
{"prompt": "...", "chosen": "好的回复", "rejected": "差的回复"}
```

#### 优化目标

$$\mathcal{L}_{DPO} = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]$$

直觉理解：提高 $\pi_\theta$ 对 chosen 回复的相对概率，降低对 rejected 回复的相对概率。

#### 优点

| 维度 | RLHF (PPO) | DPO |
|------|-----------|-----|
| 需要 RM | ✅ | ❌ |
| 训练时模型数 | 4 个 | 2 个（$\pi_\theta$ + 冻结的 $\pi_{ref}$） |
| 训练方式 | 强化学习 | 监督学习（分类损失） |
| 稳定性 | 低 | 高 |
| 实现复杂度 | 高 | 低（~100 行代码） |
| 显存 | 极高（~4×） | 中等（~2×） |

#### 局限

- **离线训练**：DPO 只在预先收集的偏好数据上训练，不能像 PPO 那样在训练中"探索"新回复
- **依赖 Bradley-Terry 假设**：如果人类偏好存在不可传递性（A > B, B > C 但 C > A），DPO 的数学基础不再严格成立
- **$\beta$ 敏感**：$\beta$ 控制距离参考模型的远近，太小则过拟合偏好数据，太大则效果甚微
- **数据覆盖依赖**：效果受限于偏好数据的 prompt 覆盖范围

#### 代码示例（TRL 库的 DPOTrainer）

```python
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
ref_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")  # 冻结

# LoRA + DPO（显存友好）
peft_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])

dpo_args = DPOConfig(
    output_dir="./dpo_output",
    per_device_train_batch_size=4,
    learning_rate=5e-6,
    beta=0.1,               # KL 惩罚系数（核心超参数）
    max_length=1024,
    max_prompt_length=512,
    num_train_epochs=1,
    bf16=True,
)

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=dpo_args,
    train_dataset=dataset,     # 需含 prompt, chosen, rejected 三字段
    tokenizer=tokenizer,
    peft_config=peft_config,
)
dpo_trainer.train()
```

> 📊 **实践经验**：用 rejection sampling（拒绝采样）缓解 DPO 的离线弱点——先用当前模型生成一批新回复，用 RM 打分后筛选高质量对再做 DPO，模拟在线探索。

---

### 3.3 DPO 改进变体

| 方法 | 解决什么问题 | 核心改进 | 参考模型 |
|------|------------|---------|---------|
| **ORPO** | DPO 需要先 SFT 再 DPO 两阶段训练 | SFT + 偏好优化联合训练，单阶段完成 | ❌ 不需要 |
| **SimPO** | DPO 需要参考模型（2×显存），且有长度偏差 | 用序列平均对数概率作为隐式奖励，长度归一化 | ❌ 不需要 |
| **Iterative DPO** | DPO 缺乏在线探索 | 多次交替"生成新回复 → 收集偏好 → 训练" | ✅ 需要 |

#### ORPO 代码示例

```python
from trl import ORPOTrainer, ORPOConfig

orpo_config = ORPOConfig(
    output_dir="./orpo_output",
    beta=0.1,               # ORPO 的偏好强度系数
    max_length=1024,
    learning_rate=1e-5,
)
orpo_trainer = ORPOTrainer(
    model=model,             # 只需要一个模型（无参考模型）
    args=orpo_config,
    train_dataset=dataset,
    tokenizer=tokenizer,
)
orpo_trainer.train()
```

#### SimPO 代码示例

```python
from trl import CPOTrainer  # SimPO 在 TRL 中通过 CPOTrainer 实现

simpo_config = CPOConfig(
    output_dir="./simpo_output",
    beta=2.0,               # SimPO 用较大的 beta
    simpo_gamma=0.5,         # margin 目标间距
    cpo_alpha=0.0,           # CPO-alpha=0 等价于纯 SimPO
)
```

---

### 3.4 GRPO（组相对策略优化）

#### 解决什么问题

前面的方法（RLHF / DPO）都依赖**人类偏好**来定义"好"——但对于推理任务（数学、编程），"好不好"是可以客观验证的：答案对不对、代码能不能通过测试。GRPO 利用这种**可验证的硬奖励**来做强化学习，不需要人类标注，也不需要 Critic 网络。

#### 用什么数据

不需要人类偏好数据。需要：
- 能自动验证的 question-answer 对（如数学题及其正解）
- 或能运行测试的编程题
- 对同一 prompt 采样多组回复 $\{y_1, ..., y_G\}$

#### 优化目标

对每个 prompt 采样 $G$ 个回复，用组内标准化计算 advantage：

$$\hat{A}_i = \frac{r_i - \text{mean}(\{r_1, ..., r_G\})}{\text{std}(\{r_1, ..., r_G\})}$$

然后应用带 clip 约束的策略梯度更新（类似 PPO，但用组内 baseline 替代 Critic 网络）：

$$J_{GRPO}(\theta) = \mathbb{E} \left[ \frac{1}{|y|} \sum_{t} \min\left( \frac{\pi_\theta(y_t|...)}{\pi_{\theta_{old}}(y_t|...)} \hat{A}_i, \ \text{clip}(\dots) \hat{A}_i \right) - \beta \cdot D_{KL}(\pi_\theta \| \pi_{ref}) \right]$$

**直观理解**：对同一个数学题，采样 8 个回复。正确的回复获得 positive advantage（模型被鼓励），错误的回复获得 negative advantage（模型被抑制）。无需人类标注"哪个回推理得更好"——答案是否正确本身就提供了偏好信号。

#### 优点

- **无 Critic**：省去一半的 RL 显存（vs PPO）
- **可验证奖励**：比人类偏好更精确、更便宜、更易扩展
- **推理能力显著提升**：DeepSeek-R1 用 GRPO 将 AIME 2024 从 15.6% 提升至 71.0%（pass@1）
- **涌现反思行为**：训练中模型自发学会了"等等，让我重新检查一下"等行为

#### 局限

- **需要可验证的奖励**：对写作质量、创造力等无法自动验证的任务不适用
- **组大小 $G$ 的选择是超参数**：$G$ 太小（<4）baseline 不稳定；$G$ 太大（>64）计算成本线性增长
- **未经过人类偏好对齐的模型可能语言变得"机械"**（如过度使用 `<think>` 标签）

#### GRPO 在 DeepSeek-R1 中的应用

```
阶段1: DeepSeek-R1-Zero（纯 RL，无 SFT）
  └── GRPO 训练 → 推理能力显著提升，但语言混杂
阶段2: DeepSeek-R1（冷启动 SFT + GRPO）
  └── 少量推理链数据做 SFT → GRPO → 通用 SFT → 全场景 RL
```

---

### 3.5 Constitutional AI / RLAIF

#### 解决什么问题

RLHF 的标注成本（每条 $0.1~$1，数万对需数千美元）和可扩展性瓶颈。**用 AI 代替人类来提供偏好反馈**。

#### 核心流程

```
RLAIF 流程:
  1. SFT 模型对 prompt 生成 K 个候选回复
  2. 调用强大的 AI 模型（如 GPT-4）对回复进行评分或排序
  3. 用 AI 生成的偏好数据做 DPO 或训练 RM 后做 PPO

Constitutional AI (Anthropic 的改进版本):
  阶段1 (监督): 模型生成回复 → 根据宪法治则自我批评 → 修改回复
  阶段2 (RL): 用 AI 根据宪法比较回复 → 训练 RM → PPO
```

**宪法原则示例**：
```
"请选择更少包含有害、非法、暴力或性内容的回复。"
"请选择更能保护用户隐私的回复。"
```

#### 优点

| 维度 | 人类标注 (RLHF) | AI 标注 (RLAIF) |
|------|----------------|-----------------|
| 成本 | 高（$0.1~$1/条） | 低（API 调用费用） |
| 速度 | 慢（数周） | 快（数小时） |
| 一致性 | 低（不同标注者标准不同） | 高（同一模型标准一致） |
| 可扩展性 | 差 | 好 |

#### 局限

- **AI 偏差放大**：AI 标注者自身的偏差（如偏好更长、更"礼貌"的回复）会被学习放大
- **自我偏好循环**：同源模型间的"近亲偏好"导致盲区被放大
- **宪法设计的挑战**：宪法原则由人设计，设计者的价值观会被系统性地嵌入

---

## 4. Evaluation 工具

### 4.1 EleutherAI lm-evaluation-harness

#### 解决什么问题

大语言模型需要在**多个维度**上被系统评估（知识、推理、安全、指令遵循等），而手工评估存在三大核心问题：
1. **Prompt 格式不一致** → 结果不可比
2. **随机性不可控** → 结果不可复现
3. **指标计算不统一** → 结果不可信

#### 用什么数据

标准化的 Benchmark 数据集（200+ 预定义任务）：

| Benchmark | 样本数 | 评估维度 | 评估模式 |
|-----------|--------|---------|---------|
| MMLU | ~15,908 | 57 学科知识广度 | Log-likelihood（多项选择） |
| GSM8K | ~8,500 | 数学推理 | 生成式（答案提取 + Exact Match） |
| HellaSwag | ~10,000 | 常识推理 | Log-likelihood（句子补全） |
| HumanEval | 164 | 代码生成 | 生成式（运行测试用例） |
| TruthfulQA | 817 | 诚实性 | Log-likelihood（二选判断） |
| ARC-Challenge | ~2,590 | 科学推理 | Log-likelihood（多项选择） |

#### 两种评估模式

**Log-likelihood 评估**（高效、确定性）：
$$\text{score}(a_i) = \sum_{t=1}^{|a_i|} \log P(a_i^{(t)} | c, a_i^{(<t)}; \theta)$$

- 适用：多项选择任务（MMLU、ARC、HellaSwag）
- 优点：一次前向传播，确定性结果
- 局限：与真实使用场景不一致

**生成式评估**（更真实）：
$$\hat{y} = \arg\max_{y} P(y | x; \theta)$$

- 适用：数学推理（GSM8K）、代码生成（HumanEval）
- 优点：接近真实使用场景
- 局限：慢 10-100×，受采样策略影响

#### 命令行示例

```bash
# 基础评估（HF 模型）
lm_eval \
  --model hf \
  --model_args pretrained=mistralai/Mistral-7B-v0.1 \
  --tasks mmlu \
  --device cuda:0 \
  --batch_size 4 \
  --num_fewshot 5 \
  --seed 42

# 多任务联合评估（使用 vLLM 后端加速）
lm_eval \
  --model vllm \
  --model_args pretrained=Qwen/Qwen2.5-7B-Instruct,tensor_parallel_size=1 \
  --tasks "mmlu,gsm8k,hellaswag,arc_challenge,truthfulqa" \
  --batch_size auto \
  --num_fewshot 5 \
  --output_path ./results/eval.json \
  --log_samples
```

#### 预期输出解读

```
| Task         | Metric    | Value | Stderr |
|------------- |-----------|------|--------|
| mmlu         | acc       | 0.705| 0.0040 | ← 知识广度
| gsm8k        | exact_match| 0.645| 0.0130 | ← 数学推理
| hellaswag    | acc_norm  | 0.810| 0.0039 | ← 常识推理
| arc_challenge| acc_norm  | 0.600| 0.0090 | ← 科学推理
| truthfulqa   | acc       | 0.520| 0.0150 | ← 诚实性（越低越易幻觉）
```

**综合诊断**：单一指标无法反映模型全貌。例如 HellaSwag 81% 说明常识推理优秀，但 TruthfulQA 52% 说明模型有"自信地胡说"的倾向——需要做对齐来改善。

#### 优点

- **标准化**：统一 prompt 模板、seed 控制、指标计算——结果可比、可复现
- **确定性**：Log-likelihood 模式提供绝对确定的结果（不涉及采样）
- **扩展性**：200+ 预定义任务，支持通过 YAML 自定义新任务
- **多后端支持**：HF / vLLM / OpenAI / Anthropic / GGUF 等

#### 局限

| 局限 | 说明 | 应对 |
|------|------|------|
| Benchmark 污染 | 预训练数据可能包含测试集 | 使用动态基准（MMLU-Pro, LiveBench） |
| 评估维度偏窄 | 偏重知识记忆和基础推理 | 补充 Agent、对话、工具使用评估 |
| 与人类偏好脱节 | Benchmark 排名 ≠ 用户满意度 | 结合人工评估（Chatbot Arena） |
| 聚合指标的信息损失 | 辛普森悖论 → 总分误导 | 始终检查子任务级别分数 |

#### 自定议评估任务（YAML 示例）

```yaml
# custom_task.yaml
task: my_custom_qa
dataset_path: my_company/my_dataset
output_type: generate_until
doc_to_text: "问题：{{question}}\n答案："
doc_to_target: "{{answer}}"
generation_kwargs:
  until: ["\n"]
  max_gen_tokens: 128
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
```

```bash
# 使用自定义 YAML 任务
lm_eval --model hf --model_args pretrained=Qwen/Qwen2.5-7B-Instruct \
  --tasks custom_task.yaml --num_fewshot 3
```

---

### 4.2 LLM-as-Judge 评估模式

#### 解决什么问题

对于开放式对话、写作、创意生成等任务，自动化指标（BLEU、ROUGE）与人类判断的低相关性使得评估不可靠。LLM-as-Judge 用强大的语言模型（GPT-4、Claude）作为裁判来评估回复质量。

#### 评估 Prompt 模板

```
请以公正的裁判身份评估 AI 助手的回复质量。
考虑因素：有帮助性、相关性、准确性、深度、创造性。

[问题]
{question}

[回复开始]
{answer}
[回复结束]

请对回复从 1 到 10 进行评分。
```

#### 代码示例（Python）

```python
import numpy as np

def llm_judge(question, answer_a, answer_b, judges=["gpt-4", "claude-3"]):
    """多裁判、位置交换来减轻偏差"""
    scores = []
    for judge in judges:
        # 原始顺序
        score_orig = query_judge(judge, question, answer_a, answer_b)
        # 交换位置
        score_swapped = query_judge(judge, question, answer_b, answer_a)
        # 取两个位置的平均（消去位置偏差）
        scores.append((score_orig + (1 - score_swapped)) / 2)
    return np.mean(scores)  # 多裁判平均
```

#### 已知偏差与缓解

| 偏差类型 | 表现 | 缓解方法 |
|---------|------|---------|
| 位置偏差 | 偏爱先看到的回复 | 交换 A/B 位置取平均 |
| 长度偏差 | 偏爱更长的回复 | 评分指令中明确"不考虑长度" |
| 自我增强偏差 | GPT-4 偏爱 GPT-4 风格 | 使用多个不同裁判模型 |
| 格式化偏差 | Markdown/列表影响评分 | 统一格式后再提交 |

---

## 5. 综合管线示例

以下是一个完整的"微调 → 对齐 → 评估"管线的逻辑伪代码，展示三种工具的协同使用：

```python
# ===== 阶段 1: SFT（指令微调） =====
# 工具: HuggingFace Trainer + LoRA
# 数据: 指令-回复对 (ChatML 格式)
# 目标: 让基座模型学会遵循指令

model = load_base_model("Qwen/Qwen2.5-7B")
model = inject_lora(model, r=8, target_modules=["q_proj", "v_proj"])
sft_trainer = SFTTrainer(model=model, dataset=sft_data)
sft_trainer.train()
model.save_pretrained("./checkpoints/sft_model")


# ===== 阶段 2: DPO（偏好对齐） =====
# 工具: TRL DPOTrainer + LoRA
# 数据: 偏好对 (prompt, chosen, rejected)
# 目标: 让模型输出更符合人类偏好

model = load_base_model("Qwen/Qwen2.5-7B")
model.load_adapter("./checkpoints/sft_model")  # 加载 SFT 权重
ref_model = load_base_model("Qwen/Qwen2.5-7B")  # 冻结的参考模型

dpo_trainer = DPOTrainer(
    model=model, ref_model=ref_model,
    beta=0.1, dataset=preference_data
)
dpo_trainer.train()


# ===== 阶段 3: 评估（系统评估） =====
# 工具: lm-evaluation-harness
# 数据: 标准化 Benchmark
# 目标: 评估模型在知识、推理、安全等维度的表现

# 命令行执行:
# lm_eval --model hf \
#   --model_args pretrained=./final_model \
#   --tasks "mmlu,gsm8k,hellaswag,truthfulqa" \
#   --num_fewshot 5 --seed 42

# 根据评估结果决定是否：
# - 补充更多数据（如 GSM8K 分数低 → 增加数学数据）
# - 调整对齐策略（如 TruthfulQA 分数低 → 补充诚实性偏好对）
# - 回到阶段 2 迭代
```

> 🔗 **完整的端到端管线指南、常见失败模式诊断清单、超参数调优建议 → 详见《Fine-tuning 技术笔记》第 17 节。**

---

## 参考资源

| 类别 | 工具 / 框架 | 用途 |
|------|------------|------|
| 微调 | [HuggingFace Transformers](https://github.com/huggingface/transformers) | 模型加载、训练、推理 |
| 微调 | [PEFT](https://github.com/huggingface/peft) | LoRA / QLoRA / DoRA 等参数高效微调 |
| 微调 | [TRL](https://github.com/huggingface/trl) | SFTTrainer / DPOTrainer / PPOTrainer |
| 微调 | [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) | 4-bit 量化（QLoRA 的基础） |
| 微调 | [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) | 一站式微调框架，支持 DPO/ORPO/SimPO |
| 对齐 | [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | RLHF 分布式训练框架 |
| 对齐 | [veRL](https://github.com/volcengine/verl) | 字节跳动的 RLHF 框架 |
| 评估 | [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | 标准化 LLM 评估框架 |
| 评估 | [OpenCompass](https://opencompass.org.cn) | 上海 AI Lab 的评估平台（支持多模态） |
| 评估 | [HELM](https://crfm.stanford.edu/helm/) | Stanford 的全面评估框架 |

---

> **本文档是 Project2_TaskB_Survey 的一部分**，与其他两篇文档形成互补：
> - [`lm_eval_example_commands.md`](./lm_eval_example_commands.md)——lm-evaluation-harness 的命令行参考
> - 本文——微调/对齐/评估工具的横向对比与代码实践
>
> **核心参考笔记**：
> - 《Fine-tuning 技术笔记》——微调方法、数据格式、超参数、资源估算
> - 《Alignment 技术笔记》——RLHF/DPO/GRPO 的数学推导、方法对比、选型指南
> - 《Evaluation Harness 技术深度解析》——Benchmark 详解、统计检验、局限性分析
