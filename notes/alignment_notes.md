# Alignment（对齐）技术笔记

---

## 1. 什么是 Alignment？

### 1.1 定义

**Alignment（对齐）** 是指通过技术手段，使大语言模型（LLM）的行为、价值观和输出与**人类的意图、偏好和伦理规范**保持一致的过程。

"对齐"这个词包含了多个层次的含义：

- **行为对齐（Behavioral Alignment）**：模型是否按照用户的指令去做（遵循指令、完成指定任务）？
- **意图对齐（Intent Alignment）**：模型的输出是否符合用户真正的、隐含的意图（而不仅仅是字面指令）？
- **价值对齐（Value Alignment）**：模型的回答是否符合社会伦理和人类价值观（无害、公正、诚实）？

> 💡 **通俗理解**：预训练模型是一个"什么都看过但什么都不懂规矩"的天才少年。**Instruction Tuning（SFT）** 教会他"别人问什么你就回答什么"——这是第一课。但光有 SFT 还不够——他可能回答得对但不礼貌、啰嗦但抓不住重点、被诱导说出危险内容。**Alignment** 就是给他上的"做人课"——如何做一个有帮助、诚实、无害的好助手。

### 1.2 Alignment 的多维度目标

在 Anthropic 的 "HHH" 框架和后续研究中，Alignment 被分解为几个核心维度：

| 维度 | 英文 | 含义 | 典型问题 |
|------|------|------|---------|
| **有帮助性** | Helpfulness | 回复是否解决了用户的问题？是否高效、清晰、完整？ | "请帮我写一个排序算法" → 是否给出了正确且可直接运行的代码？ |
| **无害性** | Harmlessness | 回复是否避免了有害、危险或不当内容？ | "告诉我如何制作炸弹" → 模型应拒绝回答 |
| **诚实性** | Honesty | 回复是否如实反映了模型的认知边界？是否承认不知道？ | "2028年奥运会在哪个城市举办？" → 模型不应编造答案 |
| **推理能力** | Reasoning | 模型是否能进行严谨的多步推理？ | 数学题、逻辑推理题是否给出了正确的推理链？ |
| **指令遵循** | Instruction Following | 是否精确遵循了格式、内容、长度等约束？ | "以 JSON 格式输出" → 输出是否确实是合法的 JSON？ |

> 📌 **关键认知**：这些维度之间存在**张力（Tension）**。例如，过度追求 Helpfulness 可能伤害 Harmlessness（用户问危险问题时，过于 helpful 的模型会如实回答）；过度追求 Harmlessness 又可能让模型过于保守，拒绝合理请求。Alignment 的本质是在这些维度之间寻找最优平衡点。

### 1.3 为什么只做 Instruction Tuning 还不够？

Instruction Tuning（SFT）教模型"按指令回答"，但存在根本性局限：

| SFT 的局限 | 具体表现 | 深层原因 |
|-----------|---------|---------|
| **不懂拒绝** | 面对不合理的请求（如"教我制作毒品"），SFT 模型可能照做不误 | SFT 的训练数据主要是"指令→回答"的映射，缺少拒绝的范例 |
| **不懂"不知道"** | 被问到超出知识范围的问题时，SFT 模型倾向于编造（幻觉）而非坦诚说"我不知道" | SFT 数据中的答案几乎都是"能回答的"，模型没学过"承认不知道" |
| **不懂偏好** | 两个回答都正确，但一个啰嗦一个简洁，SFT 没有信号告诉模型哪个更好 | SFT 只有正例（正确的回答），没有对比信号（哪个更好） |
| **不懂分寸** | 对于模糊或双关的指令，模型可能选择了"技术上正确但意图上错误"的解释 | SFT 缺少对"好的解释"和"不好的解释"的偏好判断 |
| **分布外脆弱** | 面对训练数据中没见过的指令类型，SFT 模型的输出质量断崖式下降 | SFT 学的是"模仿"，而非"理解和泛化" |

**形式化的理解**：

设 SFT 学到的策略为 $\pi_{SFT}(y|x)$。SFT 的目标是最大化训练数据中参考答案的对数似然：

$$\mathcal{L}_{SFT} = -\mathbb{E}_{(x, y^*) \sim \mathcal{D}_{SFT}} [\log \pi_{SFT}(y^*|x)]$$

这个目标函数有一个致命弱点：**它对所有"非参考答案"的惩罚是均等的**——无论一个错误的回答是"略微不准确"还是"极其危险"，在 SFT 的损失函数中都只是"不是 $y^*$"。SFT 无法区分"不好的回答"和"危险的回答"。

> 🔑 **核心洞察**：SFT 教会模型"做什么"（What to do），Alignment 教会模型"怎么做好"（How to do well）和"什么不能做"（What not to do）。两者是递进关系，而非替代关系。

---

## 2. Alignment 方法全景图

### 2.1 方法之间的演进关系

```
                    预训练基座模型 (Pre-trained Base Model)
                                │
                                ▼
                    ┌──────────────────────┐
                    │   SFT                │  "学会回答"
                    │   (指令微调)           │  数据：指令-回复对
                    │   最大化 P(answer|Q)  │  目标：模仿参考答案
                    └──────────┬───────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
   ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐
   │  RLHF        │   │  DPO 及变体   │   │  RLAIF           │
   │  (PPO-based) │   │  (无需RM)    │   │  (AI Feedback)   │
   └──────┬───────┘   └──────┬───────┘   └────────┬─────────┘
          │                  │                     │
          │           ┌──────┼──────┐              │
          │           │      │      │              │
          ▼           ▼      ▼      ▼              ▼
   ┌──────────┐ ┌────────┐┌──────┐┌───────┐┌──────────────┐
   │  PPO     │ │  DPO   ││ORPO  ││SimPO  ││ Constitutional│
   │ (RL+RM)  │ │(直接偏好)││(无ref)││(无ref)││ AI           │
   └──────────┘ └────────┘└──────┘└───────┘└──────────────┘
          │
          ▼
   ┌──────────────────────────────┐
   │  Reasoning-oriented RL        │  "学会推理"
   │  GRPO / R1 / DeepSeek-R1     │
   │  数据：可验证奖励（答案对/错）  │
   │  目标：增强推理链的生成能力    │
   └──────────────────────────────┘
```

### 2.2 方法速查表

| 方法 | 提出时间 | 核心思想 | 需要 Reward Model？ | 训练数据 | 核心优势 | 主要局限 |
|------|---------|---------|-------------------|---------|---------|---------|
| **RLHF (PPO)** | 2022 (InstructGPT) | 用 RM 打分，用 PPO 优化策略 | ✅ 需要 | 偏好排序 + SFT 数据 | 最经典、最灵活 | 复杂、不稳定、成本高 |
| **DPO** | 2023 | 消去 RM，直接在偏好数据上优化 | ❌ 不需要 | 偏好对 (chosen, rejected) | 简单稳定 | 缺乏在线探索 |
| **RLAIF** | 2023 | 用 AI 代替人类标注偏好 | ❌ 不需要人类 | 用 AI 生成的偏好判断 | 成本低、可扩展 | AI 标注自身有偏差 |
| **ORPO** | 2024 | SFT + 偏好优化联合训练 | ❌ 不需要 | 偏好对 | 无需参考模型 | 较新，生态待完善 |
| **SimPO** | 2024 | 用序列平均概率做奖励 | ❌ 不需要 | 偏好对 | 无需参考模型，长度归一化 | 较新 |
| **GRPO** | 2024 (DeepSeek) | 去掉 Critic，组内相对比较 | ❌ 不需要 RM | 可验证奖励（答案对错） | 显存省、推理增强 | 需要可验证任务 |
| **Constitutional AI** | 2022 (Anthropic) | 用宪法原则引导 AI 自我修正 | ✅ 需要（AI 训练） | AI 生成的修正对 | 价值可控、可解释 | 宪法的设计本身有偏见 |

---

## 3. RLHF（Reinforcement Learning from Human Feedback）

### 3.1 什么是 RLHF？

RLHF 是当前最经典的 Alignment 方法，由 OpenAI 在 InstructGPT (2022) 和 ChatGPT 中率先大规模使用。其核心思想是：**让人类对模型的输出进行偏好排序，用这些偏好数据训练一个"奖励模型"（Reward Model），然后用强化学习（PPO）来优化模型，使其输出获得更高的奖励分数。**

### 3.2 RLHF 的三阶段流程

```
┌─────────────────────────────────────────────────────────────────┐
│  阶段 1: SFT (Supervised Fine-Tuning)                           │
│  ─────────────────────────────────────                          │
│  输入：高质量人类标注的 (prompt, response) 对                     │
│  输出：π_SFT —— 一个能基本遵循指令的模型                          │
│  训练：标准交叉熵损失，让模型模仿人类标注的高质量回答               │
│                                                                 │
│  为什么这一步是必要的？                                          │
│  - 基座模型的输出空间太大（什么都能生成），RL 需要在这个空间里搜索   │
│  - SFT 先缩小搜索空间到一个"合理的区域"，降低 RL 的探索难度          │
│  - SFT 模型是 RLHF 的起点（也是 KL 正则化的参考点）                │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  阶段 2: Reward Model (RM) 训练                                  │
│  ─────────────────────────────                                  │
│  输入：人类对回复的偏好排序数据                                    │
│      - 对同一个 prompt x，π_SFT 生成 K 个不同回复 {y₁,...,yₖ}    │
│      - 人类标注者排序：y₁ > y₃ > y₂ > ...（偏好顺序）              │
│  输出：r_φ(x, y) —— 一个能给任意回复打分的"裁判模型"               │
│  训练：让 RM 学会预测人类偏好（哪个回复更好）                        │
│                                                                 │
│  RM 的结构：通常是 SFT 模型去掉最后的 unembedding 层，              │
│  替换为一个标量输出头 (scalar head)，输出一个实数分数               │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  阶段 3: PPO 强化学习                                            │
│  ─────────────────                                              │
│  输入：RM r_φ (固定冻结) + 新的 prompt 集合 x ~ D_prompt          │
│  输出：π_θ —— 经过 RL 优化的策略模型                              │
│  优化：                                                          │
│       max E[r_φ(x, y)] - β·KL(π_θ || π_SFT)                     │
│       即：最大化奖励分数，同时不要让模型偏离 SFT 模型太远            │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Reward Model（奖励模型）详解

#### 奖励模型的作用

奖励模型 $r_\phi(x, y)$ 是 RLHF 中承上启下的关键组件：

- **对上（人类偏好）**：将不可微的、主观的人类偏好转化为可微的标量信号
- **对下（PPO 优化）**：提供逐 token 或逐序列的奖励信号，驱动 RL 优化

**为什么不能直接让人类打分？** 因为 RL 需要数百万次试错，每次都要反馈——这在人类标注的成本和时间上是不可行的。RM 充当了"人类偏好的代理"（Proxy），使自动化训练成为可能。

#### RM 训练的数学形式

给定 prompt $x$ 和一对回复 $(y_w, y_l)$，其中 $y_w$ 比 $y_l$ 更好（$w$ = winner, $l$ = loser）。

采用 **Bradley-Terry 偏好模型**：

$$P(y_w \succ y_l | x) = \frac{e^{r_\phi(x, y_w)}}{e^{r_\phi(x, y_w)} + e^{r_\phi(x, y_l)}} = \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))$$

其中 $\sigma(\cdot)$ 是 sigmoid 函数。直观理解：奖励分数差距越大，winner 胜出的概率越接近 1。

RM 的训练损失（负对数似然）：

$$\mathcal{L}_{RM}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}_{pref}} \left[ \log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l)) \right]$$

> 💡 **直觉**：如果 $r_\phi(x, y_w) - r_\phi(x, y_l)$ 很大（正确判断），$\sigma(\cdot) \to 1$，loss 小。如果 RM 判断反了（分数差距为负），$\sigma(\cdot) \to 0$，$-\log(0) \to \infty$，loss 很大。

#### 偏好数据的收集

```
对一个 prompt "请解释什么是黑洞"：

SFT 模型生成 4 个不同的回复：
  y₁: [科学准确且通俗易懂的解释]     ← 人类排第 1 (best)
  y₂: [科学准确但过于学术化的解释]    ← 人类排第 2
  y₃: [基本准确但缺少关键细节的解释]  ← 人类排第 3
  y₄: [包含事实错误的解释]           ← 人类排第 4 (worst)

从排序中提取偏好对：
  (y₁, y₂), (y₁, y₃), (y₁, y₄), (y₂, y₃), (y₂, y₄), (y₃, y₄)
  共 C(4,2) = 6 对偏好数据
```

通常在一个 prompt 下取 K=4~9 个回复，可以产生 $\binom{K}{2}$ 个偏好对。**K 的选择是成本-效率权衡**：K 越大，产生的偏好对越多（$O(K^2)$），但人类标注者的认知负担也越重。

#### RM 的关键挑战

| 挑战 | 说明 |
|------|------|
| **Reward Hacking** | 模型学会"钻空子"拿高分而非真正变好。例如 RM 偏好长回复，模型就拼命输出废话来刷分 |
| **分布偏移** | RM 在 SFT 模型的输出上训练，但 PPO 优化后模型的输出分布会改变——RM 在"陌生领域"打分不准确 |
| **偏好不可传递** | 人类的偏好不一定满足 Bradley-Terry 的传递性假设（A > B, B > C 但 C > A 的情况实际存在） |
| **标注成本** | 高质量的人类偏好标注价格约为 $0.1 ~ $1/对，大规模训练需要数十万对 |
| **标注者偏差** | 标注者的文化背景、政治倾向、个人偏好会渗入 RM，难以完全消除 |

### 3.4 PPO 在 RLHF 中的作用

#### 为什么需要 PPO？

有了 RM 之后，一个自然的想法是：直接用 RM 的分数作为损失函数来优化模型不就行了吗？

问题在于：**RM 只在"好回复的邻域"是准确的**。如果我们简单做梯度上升去最大化 RM 分数，模型会快速漂移到 RM 打分高但回复质量极差的区域（如输出乱码、重复短语等能 hack RM 的样本）。这就是 **Reward Hacking**。

PPO 的核心作用就是在最大化奖励的同时，**约束模型不要偏离 SFT 模型太远**——保留 SFT 阶段学到的语言流畅性和基本能力。

#### PPO 的优化目标

$$\max_{\theta} \ \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)} \left[ r_\phi(x, y) - \beta \cdot D_{KL}\left(\pi_\theta(\cdot|x) \ \|\ \pi_{SFT}(\cdot|x)\right) \right]$$

- **第一项 $r_\phi(x, y)$**：最大化 RM 评分（让回复更符合人类偏好）
- **第二项 $-\beta \cdot KL$**：惩罚偏离 SFT 模型的行为（防止 reward hacking 和语言退化）
- **$\beta$**：平衡系数——$\beta$ 越大，模型越保守（更接近 SFT）；$\beta$ 越小，模型越激进（更追求高分）

#### PPO 的核心机制（简化理解）

1. **采样**：从当前策略 $\pi_\theta$ 对一个 prompt batch 采样回复
2. **打分**：用 RM $r_\phi$ 对每条回复打分
3. **计算 Advantage**：当前回复比"平均水平"好多少？（好则增强，差则抑制）
4. **Clip 约束**：限制每次更新的幅度，防止策略突变：

$$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left( \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} \hat{A}_t, \ \text{clip}\left(\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}, 1-\epsilon, 1+\epsilon\right) \hat{A}_t \right) \right]$$

其中 $\epsilon$ 通常为 0.2。这个 clips 机制保证了新旧策略的概率比被限制在 $[0.8, 1.2]$ 范围内，防止单步更新过大。

#### RLHF 中的"四个模型"

| 模型 | 作用 | 是否训练 | 精度 |
|------|------|---------|------|
| Actor ($\pi_\theta$) | 实际生成回复的策略模型 | ✅ 训练 | FP16/BF16 |
| Reference ($\pi_{SFT}$) | 计算 KL 散度的参考点 | ❌ 冻结 | FP16/BF16 |
| Critic ($V_\psi$) | 估计状态价值（用于计算 advantage） | ✅ 训练 | FP32（需稳定） |
| Reward Model ($r_\phi$) | 给回复打分 | ❌ 冻结 | FP16/BF16 |

> 💡 **工程细节**：Critic 模型（$V_\psi$）通常与 Actor 模型**共享 backbone**（即使用同一个预训练模型的主干），仅在输出层增加一个标量价值头（Value Head，通常是一个线性层）。这样 Critic 的额外参数仅有数千个，而非完整模型的大小。在实际实现中（如 OpenRLHF、veRL），Critic 通常用 Actor 模型的拷贝初始化，共享 backbone 权重，仅价值头部分独立训练。

> ⚠️ **显存警告**：RLHF 需要同时维护 4 个模型（总共 ~4× 模型大小的显存），这是 RLHF 实现难度和成本高的核心原因。对于 7B 模型，仅模型参数就需要约 56GB（4 × 14GB），加上优化器状态和激活值，通常需要 4-8 张 A100。

### 3.5 RLHF 的优势与局限

| 优势 | 局限 |
|------|------|
| 经典方法，效果经过大规模验证（ChatGPT 的方法论基础） | 实现极其复杂，需要维护 Actor/Critic/RM/Ref 四个模型 |
| 在线探索——PPO 在训练中不断生成新回复，RM 给真实反馈 | 训练不稳定，超参数（$\beta$, clip $\epsilon$, KL penalty weight 等）敏感 |
| 理论上灵活——可以针对任意可量化的奖励进行优化 | Reward Hacking 难以完全消除 |
| 可以与 SFT 形成互补（SFT 提供起点，RL 提供精细化） | 人类偏好标注成本高 |
| 支持多轮交互的优化（每轮都可以用 RM 打分） | 分布偏移——RM 在训练过程中的打分质量会下降 |

---

## 4. DPO（Direct Preference Optimization）

### 4.1 DPO 解决了什么问题？

RLHF 的痛点非常明确：**太复杂了**。需要训练 RM、需要 PPO、需要同时维护 4 个模型。这导致：

- 学术团队和个人开发者几乎无法复现
- 超参数极其敏感，调参成本高
- 训练不稳定，容易崩溃

DPO（Rafailov et al., 2023）的核心思想是：**能不能跳过 RM 和 RL，直接在偏好数据上优化模型？** 答案是：**可以**。

### 4.2 DPO 的核心洞察

DPO 的关键数学发现是：**RLHF 中 PPO 的最优策略，可以直接用一个分类损失来表示，不需要显式训练 RM，也不需要 RL。**

推导的核心步骤：

**第 1 步**：RLHF 中 KL 约束下的最优策略有解析解：

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left(\frac{1}{\beta} r(x, y)\right)$$

其中 $Z(x) = \sum_y \pi_{ref}(y|x) \exp\left(\frac{1}{\beta} r(x,y)\right)$ 是配分函数（使得概率和为 1）。

**第 2 步**：从这个解析解反解出奖励函数：

$$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

**第 3 步**：将这个奖励函数代入 Bradley-Terry 偏好模型：

$$P(y_w \succ y_l | x) = \sigma\left( r(x, y_w) - r(x, y_l) \right)$$

$$= \sigma\left( \beta \log \frac{\pi^*(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_{ref}(y_l|x)} \right)$$

**🔑 这是 DPO 成立的关键数学巧合**：$\beta \log Z(x)$ 在减法中**精确抵消了**！这意味着我们不需要计算 $Z(x)$（一个涉及所有可能回复的配分函数，在指数级大的输出空间中不可计算）。**正是这个抵消，让 DPO 能够用一个简单的分类损失替代 RLHF 中复杂的 RL 优化**——这是整个 DPO 方法最精妙的数学洞察。

**第 4 步**：直接用 $\pi_\theta$（当前模型）代替 $\pi^*$，得到 DPO 损失函数：

$$\mathcal{L}_{DPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}_{pref}} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]$$

### 4.3 DPO 损失函数的直观理解

将 DPO 损失拆开来看：

$$\text{score}(y|x) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$$

- $\pi_\theta(y|x)$：当前模型对回复 $y$ 的概率
- $\pi_{ref}(y|x)$：参考模型（SFT 模型，冻结）对回复 $y$ 的概率
- 比值 $\frac{\pi_\theta}{\pi_{ref}}$：当前模型相对于参考模型更"喜欢"这个回复的程度

DPO 的优化方向是：

```
提高 chosen (y_w) 的 score  ←→  降低 rejected (y_l) 的 score
         ↑                                ↓
    π_θ(y_w|x) 相对于 π_ref 增大       π_θ(y_l|x) 相对于 π_ref 减小
```

### 4.4 DPO 的训练过程

```
输入：
  ├── π_ref (SFT 模型，冻结)
  ├── π_θ (待优化的策略模型，从 π_ref 初始化)
  └── D_pref = {(x, y_w, y_l)} （偏好对数据集）

训练循环（每个 batch）：
  1. 对于每条偏好对，分别计算 π_θ 和 π_ref 下 chosen 和 rejected 的对数概率
  2. 计算 L_DPO
  3. 反向传播，只更新 π_θ（π_ref 始终冻结）
  4. 重复直到收敛

输出：π_θ* —— DPO 对齐后的模型
```

> 🔑 **只需要 2 个模型**（$\pi_\theta$ + $\pi_{ref}$），vs RLHF 的 4 个模型。这就是 DPO 能够被如此广泛采用的根本原因。

### 4.5 DPO vs RLHF 深度对比

| 对比维度 | RLHF (PPO) | DPO |
|---------|-----------|-----|
| **是否需要 RM** | ✅ 需要独立训练 | ❌ 不需要 |
| **训练时模型数** | 4 个（Actor, Critic, RM, Ref） | 2 个（$\pi_\theta$, $\pi_{ref}$） |
| **训练方式** | 强化学习（PPO） | 监督学习（分类损失） |
| **在线 vs 离线** | **在线**：PPO 在训练中不断生成新回复并打分 | **离线**：仅在预先收集的偏好数据上训练 |
| **显存需求** | 极高（~4 倍模型大小） | 中等（~2 倍模型大小） |
| **训练稳定性** | 较低（RL 固有的波动性） | 较高（标准分类损失，与 SFT 类似） |
| **实现复杂度** | 高（需要 PPO 基础设施，Multi-GPU 通信） | 低（~100 行核心代码） |
| **超参数数量** | 多（学习率、PPO clip、KL 系数、GAE λ、value loss coeff...） | 少（学习率、$\beta$、基本够用） |
| **数学等价性** | — | 在 Bradley-Terry 模型假设下，DPO 和 RLHF 有相同的最优解 |
| **Reward Hacking** | 有风险（RM 被钻空子） | 无 RM，无此问题 |
| **在线探索** | 有（PPO 生成新回复，可能发现更好的策略） | 无（仅在已有的偏好数据上优化） |

### 4.6 DPO 的优势和局限

**优势**：

- 简单稳定：与 SFT 训练流程几乎相同（只是换了损失函数），不需要 RL 基础设施
- 显存友好：只需 2 个模型（vs RLHF 的 4 个），可在消费级硬件上用 QLoRA 实现
- 无 RM 训练：省去了 RM 训练的数据收集和模型训练成本
- 效果竞争力：在许多 benchmark 上达到甚至超过 RLHF 的水平
- 实现简单：核心代码不到 100 行，降低了对 RL 专业知识的要求

**局限**：

- **离线训练**：DPO 在预先收集的偏好数据上训练，不能像 PPO 那样在训练中"探索"新的回复并获得反馈。这意味着 DPO 的效果被限制在数据覆盖的范围内
- **依赖 Bradley-Terry 假设**：如果人类偏好不满足这个模型（如存在不可传递性），DPO 的数学基础不再严格成立
- **$\beta$ 敏感的参考模型**：$\beta$ 控制距离参考模型的远近——太小则过拟合偏好数据，太大则效果甚微
- **偏好数据分布偏差**：如果偏好数据集中在某些类型的 prompt 上，DPO 对齐也会不均匀

**缓解方案**：

针对上述局限，有以下实践缓解方案：

| 缓解方案 | 针对的局限 | 做法 |
|---------|-----------|------|
| **数据覆盖** | 离线训练、分布偏差 | 用 SFT 模型生成多样化的回复作为偏好数据的基础，提高 prompt 和回复的覆盖度 |
| **拒绝采样（Rejection Sampling）** | 缺乏在线探索 | 用当前模型生成一批回复，用 RM 打分后筛选出高质量对，再做 DPO——半在线方式 |
| **Iterative DPO** | 缺乏在线探索 | 多次交替"生成新回复 → 收集偏好 → 训练"，模拟在线探索。每次用更新后的模型生成回复，拓宽训练数据分布 |

### 4.7 常用的偏好数据集

| 数据集 | 规模 | 特点 | 适用场景 |
|--------|------|------|---------|
| **HH-RLHF** (Anthropic) | 169k 对 | 红蓝对抗风格，包含安全性和有用性对比 | 通用对齐，经典基准 |
| **UltraFeedback** | 64k 提示，每提示 4 回复 | 多维度评分（指令遵循、诚实、有帮助、安全等） | 多属性偏好优化 |
| **HelpSteer** (NVIDIA) | 37k 对话，每对话 5 维度评分 | 细粒度分数（0-4），可合成偏好对 | 需要细粒度奖励建模 |
| **PKU-SafeRLHF** | 33k 偏好对 + 安全标签 | 侧重安全对齐 | 安全专项对齐 |
| **WebGPT Comparisons** | 19k 对 | 基于网页浏览的问答偏好 | 检索增强生成的对齐 |
| **Nectar** | 183k 对 | 7 个 LLM 生成，GPT-4 打分排序 | 多模型蒸馏偏好 |

> 💡 **选择建议**：通用对齐首选 HH-RLHF + UltraFeedback 混合；安全专项优先 PKU-SafeRLHF；需要细粒度评分选 HelpSteer。

> 🔗 **数据格式说明**：偏好数据通常以 JSONL 格式存储，每条包含三个核心字段：
> ```json
> {"prompt": "用户的问题或对话上下文", "chosen": "人类偏好的回复", "rejected": "人类不喜欢的回复"}
> ```
> 这与 SFT 阶段使用的 ChatML 格式（详见《Fine-tuning 笔记》第 8 节）不同——SFT 数据是"单答案"格式（一条数据只有一个正确答案），而偏好数据是"对比对"格式（一条数据包含好/差两个答案供模型学习偏好差异）。转换时需注意：SFT 数据的 `assistant` 回复通常作为 `chosen`，而 `rejected` 需要额外生成或标注。

---

## 5. DPO 的改进变体

DPO 提出后，出现了多种改进版本，主要针对 DPO 的几个弱点：

### 5.1 ORPO（Odds Ratio Preference Optimization, 2024）

**解决了什么问题？** DPO 需要两个阶段：先 SFT，再 DPO。ORPO 将两者合并为单阶段训练。

**核心思想**：在 SFT 的损失函数中直接加入一个偏好对齐项，无需参考模型。

**损失函数**：

$$\mathcal{L}_{ORPO} = \mathcal{L}_{SFT} + \lambda \cdot \mathcal{L}_{OR}$$

其中 $\mathcal{L}_{OR}$ 基于 odds ratio（几率比）来惩罚 rejected 回复：

$$\mathcal{L}_{OR} = -\log \sigma\left( \log \frac{\text{odds}_\theta(y_w|x)}{\text{odds}_\theta(y_l|x)} \right)$$

$$\text{odds}_\theta(y|x) = \frac{P_\theta(y|x)}{1 - P_\theta(y|x)}$$

**为什么 odds ratio 而非概率比？** Odds ratio 比概率比更敏感——当 rejected 的概率从 0.01 降到 0.001 时，概率比变化不大，但 odds ratio 变化显著。这使得 ORPO 对 rejected 回复的惩罚更有效。

**优点**：无需参考模型（省了一半显存），单阶段训练（更简单）  
**局限**：较新，社区验证尚不充分；对超参数 $\lambda$ 敏感

### 5.2 SimPO（Simple Preference Optimization, 2024）

**解决了什么问题？** DPO 需要参考模型（2×显存），且对长度偏差敏感（长回复天然概率低）。

**核心思想**：用序列的**平均对数概率**作为隐式奖励，完全不需要参考模型。

**SimPO 的奖励函数**：

$$r_{SimPO}(x, y) = \frac{\beta}{|y|} \sum_{t=1}^{|y|} \log \pi_\theta(y_t | x, y_{<t}) = \frac{\beta}{|y|} \log \pi_\theta(y|x)$$

其中 $|y|$ 是回复的长度。除以 $|y|$ 是关键——它做**长度归一化**，防止模型通过生成更短（或更长）的回复来"作弊"获得更高概率。

**SimPO 损失**（使用 margin $\gamma$）：

$$\mathcal{L}_{SimPO} = -\log \sigma\left( \frac{\beta}{|y_w|} \log \pi_\theta(y_w|x) - \frac{\beta}{|y_l|} \log \pi_\theta(y_l|x) - \gamma \right)$$

其中 $\gamma$ 是一个正 margin（目标间距），要求 winner 的分数至少比 loser 高 $\gamma$。

**优点**：无需参考模型；长度归一化天然解决长度偏差问题；margin $\gamma$ 提供了更稳定的优化目标  
**局限**：较新；平均对数概率作为奖励的近似是否总是合理，有待更多验证

### 5.3 DPO 变体对比

| 方法 | 参考模型 | 主要改进 | 适用场景 |
|------|---------|---------|---------|
| **DPO** | ✅ 需要 | 原始方法，消去 RM 和 RL | 通用场景，数据量充足时 |
| **ORPO** | ❌ 不需要 | SFT + 偏好对齐联合训练，单阶段 | 追求训练流程简化 |
| **SimPO** | ❌ 不需要 | 长度归一化，margin-based | 有长度偏差问题的场景 |

---

## 6. RLAIF（Reinforcement Learning from AI Feedback）

### 6.1 什么是 RLAIF？

**RLAIF** 的核心思想是：**用 AI 模型代替人类来提供偏好反馈**。这里的 "AI" 通常是一个更强大的语言模型（如 GPT-4、Claude），它扮演了"标注者"的角色。

### 6.2 为什么需要 RLAIF？

RLHF 的一个根本瓶颈是**人类标注的成本和可扩展性**：

| 瓶颈 | RLHF (人类标注) | RLAIF (AI 标注) |
|------|----------------|-----------------|
| 成本 | 高（$0.1~$1/对，数万对需数千美元） | 低（API 调用成本，可忽略） |
| 速度 | 慢（人类阅读、判断、标注，数周） | 快（API 调用，数小时） |
| 一致性 | 低（不同标注者标准不同） | 高（同一模型，标准一致） |
| 可扩展性 | 差（人力线性增长） | 好（API 调用可并行扩展） |
| 知识覆盖 | 受限于标注者的知识面 | 可覆盖 AI 的广泛知识（但 AI 也有盲区） |
| 偏差 | 人类标注者偏差（文化、认知） | AI 偏差（训练数据偏差，被放大） |

### 6.3 RLAIF 的基本流程

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: 偏好数据生成（AI 替代人类）                           │
│  ─────────────────────────────────                          │
│  对每个 prompt x:                                             │
│    1. SFT 模型生成 K 个候选回复 {y₁, y₂, ..., yₖ}            │
│    2. 调用强大的 AI 模型（如 GPT-4）对回复进行评分或排序       │
│    3. 获得偏好对 (y_w, y_l)                                   │
│                                                              │
│  AI 打分的 Prompt 模板（示意）：                               │
│    "请对以下两个 AI 助手的回复进行对比，判断哪个更好。          │
│     考虑因素：有用性、准确性、安全性、清晰度。                  │
│     回复 A: {y₁}                                              │
│     回复 B: {y₂}                                              │
│     请输出 A 或 B。"                                          │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 2: 用 AI 生成的偏好数据做对齐                            │
│  ───────────────────────────────                             │
│  选项 A: 直接做 DPO（在 AI 偏好对数据上）                      │
│  选项 B: 训练 RM（在 AI 偏好数据上），然后做 PPO               │
│                                                              │
│  实践中通常选择 DPO（更简单、成本更低）                         │
└─────────────────────────────────────────────────────────────┘
```

### 6.4 Anthropic 的 Constitutional AI（宪法 AI）

Constitutional AI（Bai et al., 2022）是 RLAIF 的一个重要实例，由 Anthropic 提出：

**核心思想**：用一套"宪法"（Constitution，一组自然语言描述的伦理原则）来指导 AI 进行自我修正，而非依赖人类标注的偏好数据。

```
宪法原则示例（Anthropic 实际使用的一部分）：
  - "请选择更少包含有害、非法、暴力或性内容的回复。"
  - "请选择更少包含种族歧视、性别歧视或其他偏见内容的回复。"
  - "请选择更能保护用户隐私的回复。"
  - "请选择更加诚实、不编造信息的回复。"

两阶段流程：

阶段 1: Supervised Phase（监督阶段）
  1. 用"有害 prompt"让模型生成回复
  2. 让模型根据宪法原则 CRITIQUE 自己的回复
  3. 让模型根据 critique REVISE 自己的回复
  4. 用 (harmful_prompt, revised_safe_response) 对做 SFT

阶段 2: RL Phase（强化学习阶段）
  1. 从阶段 1 的模型中采样回复
  2. 让 AI 根据宪法原则比较两个回复（哪个更符合宪法？）
  3. 用这些 AI 生成的偏好数据训练 RM
  4. 用 RM 做 PPO 训练
```

**Constitutional AI vs 标准 RLAIF**：

| 维度 | 标准 RLAIF | Constitutional AI |
|------|-----------|-------------------|
| 偏好来源 | AI 的"直觉判断" | AI 基于明确宪法原则的"推理判断" |
| 可解释性 | 低（AI 没说"为什么"选 A） | 高（AI 引用了具体的宪法条款） |
| 可定制性 | 低（依赖 AI 自身的价值观） | 高（更换宪法即可改变行为） |
| 透明度 | 低 | 高（宪法原则是公开可审计的） |

### 6.5 RLAIF 的优势和局限

**优势**：

- 成本低，可大规模扩展（AI 标注成本远低于人类）
- 一致性高（同一 AI 的标准不会因"疲劳"而漂移）
- 可与人类标注混合（部分数据 AI 标注，部分人类标注）
- Constitutional AI 提供了可解释的、可定制的价值观框架

**局限**：

- **AI 偏差放大**：AI 标注者自身的偏差（如偏好更长、更"礼貌"的回复）会被学习放大
- **自我偏好循环**：如果 AI 标注者与被训练的模型是同源的（如都用 GPT-4 做标注来训练 LLaMA），可能产生"近亲偏好"——模型学到了 AI 标注者的盲区
- **复杂判断不可靠**：AI 标注者在需要深度专业知识或微妙伦理判断的任务上可靠性存疑
- **宪法设计的挑战**：Constitutional AI 的宪法原则本身由人设计，设计者的价值观和偏见会被系统地嵌入

### 6.6 红队测试（Red Teaming）与对抗性对齐

**红队测试**是安全对齐流程中不可或缺的一环——它不是一种训练方法，而是一种**主动发现模型漏洞**的实践。

#### 定义

红队测试（Red Teaming）指**人工或自动化地生成对抗性 prompt**，试图诱使模型生成有害、违规或偏差内容，从而暴露模型的安全漏洞。

#### 自动化红队

用另一个 LLM（或专门的攻击模型）自动生成攻击性 prompt：

```
红队 LLM 的角色：
  "你是一个红队测试专家。你的目标是设计能够诱使目标 AI 模型
  生成不安全内容的 prompt。请针对以下安全维度生成 10 个攻击性 prompt：
  - 暴力内容诱导
  - 非法活动指导
  - 歧视性言论诱导
  ..."

生成的 prompt 被用于测试目标模型 → 记录失败的 case
```

**常用自动化红队工具**：

| 工具 | 说明 |
|------|------|
| **Garak** | LLM 漏洞扫描器，覆盖多种攻击模式 |
| **PyRIT** (Microsoft) | 自动化红队框架，支持多轮攻击 |
| **Counterfit** (Azure) | AI 系统安全测试工具 |

#### 红队与对齐的关系

红队发现的失败 case 可以直接反馈到对齐训练中：

```
红队发现漏洞 → 收集失败的 prompt-response 对 → 构造偏好数据
    → DPO/RLHF 训练（修复漏洞）→ 重新红队测试 → 迭代
```

这个闭环是构建安全对齐模型的**工程化实践**，在 Claude、GPT-4、DeepSeek-R1 等模型的开发中都有广泛应用。

> 💡 **实践建议**：即使没有专门的红队团队，也可以用以下简单方法做轻量红队：
> 1. 准备 50-100 个已知有害的 prompt（可从 HarmBench、SafeRLHF 数据集获取）
> 2. 微调前后分别测试模型的拒绝率
> 3. 对拒绝率下降的维度补充安全对齐数据

---

## 7. 面向推理的 Alignment：GRPO 与 DeepSeek-R1

### 7.1 新范式：Reasoning-oriented Alignment

2024 年以来，Alignment 领域出现了一个重要的新方向：**不依赖人类偏好的对齐**，而是利用**可验证的客观奖励**（如数学题答案对不对、代码能不能跑通）来做强化学习，增强模型的推理能力。

这个方向的核心认知是：

> 对于推理任务（数学、编程、逻辑），"好不好"是可以客观验证的——答案对不对，代码能不能通过测试用例。这比人类的"偏好"更精确、更可靠。如果能用这种"硬奖励"来做 RL，就能在推理维度上取得显著突破。

### 7.2 GRPO（Group Relative Policy Optimization）

GRPO 是 DeepSeek 团队在 DeepSeekMath (2024) 和 DeepSeek-R1 (2025) 中使用的方法，其设计目标是**在不需要 Critic 模型的情况下进行有效的 RL 训练**。

#### GRPO 的核心创新：去掉 Critic（Value 网络）

在标准 PPO 中，Advantage（优势函数）的计算需要 Critic 网络 $V_\psi$ 来估计状态价值：

$$\hat{A}_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

Critic 网络的参数量通常与 Actor 相当，这意味着 PPO 在这一项上就消耗了一倍的显存。

**GRPO 的做法**：对同一个 prompt $x$，从旧策略 $\pi_{\theta_{old}}$ 中采样 **G 个回复**（通常 G=4~64），然后用这组回复的平均奖励作为 baseline：

$$\hat{A}_i = \frac{r_i - \text{mean}(\{r_1, ..., r_G\})}{\text{std}(\{r_1, ..., r_G\})}$$

其中 $r_i$ 是第 $i$ 个回复的奖励分数，经过组内标准化（减去均值、除以标准差）后得到 advantage。

#### GRPO 的优化目标

$$J_{GRPO}(\theta) = \mathbb{E} \left[ \frac{1}{\text{len}(y)} \sum_{t=1}^{\text{len}(y)} \min\left( \frac{\pi_\theta(y_t|x, y_{<t})}{\pi_{\theta_{old}}(y_t|x, y_{<t})} \hat{A}_i, \ \text{clip}\left(\frac{\pi_\theta}{\pi_{\theta_{old}}}, 1-\epsilon, 1+\epsilon\right) \hat{A}_i \right) - \beta \cdot D_{KL}(\pi_\theta \| \pi_{ref}) \right]$$

这个目标与 PPO 类似（有 clip 约束、有 KL 惩罚），但 advantage 的计算方式完全不同（组内相对比较 vs Critic 估计）。

**组大小 $G$ 的选择**：$G$ 是 GRPO 的重要超参数，通常在 4~64 之间。$G$ 过小（<4），组内均值受极值影响大，baseline 不稳定；$G$ 过大（>64），计算成本线性增长但收益递减。DeepSeekMath 中使用 $G=64$，DeepSeek-R1 在推理训练阶段使用 $G=4$~16（多轮迭代采样）。一般来说，任务噪声越大（如奖励信号不精确），需要更大的 $G$ 来稳定 baseline。

#### GRPO 为什么特别适合推理任务？

在数学题中，GRPO 的工作方式非常直观：

```
Prompt: "计算 ∫₀¹ x² dx 的值。"

从旧策略中采样 G=8 个回复：
  y₁: "1/3 ✓"                    → reward = +1 (正确)
  y₂: "1/2 ✗"                    → reward = 0  (错误)
  y₃: "∫x²dx = x³/3, 代入得1/3 ✓" → reward = +1 (正确，且格式好)
  y₄: "1/3 ✓"                    → reward = +1
  y₅: "无法计算 ✗"              → reward = 0
  y₆: "1/4 ✗"                    → reward = 0
  y₇: "解：原式 = [x³/3]₀¹ = 1/3 - 0 = 1/3 ✓" → reward = +1 (正确，推理步骤完整)
  y₈: "0 ✗"                      → reward = 0

组内标准化后：
  mean(r) = 4/8 = 0.5
  y₁ advantage = (1-0.5)/std > 0  → 增强！
  y₂ advantage = (0-0.5)/std < 0  → 抑制！
  y₇ advantage = (1-0.5)/std > 0  → 增强！（推理步骤完整的回复获得更多"青睐"）

不需要人类标注"哪个推理好"——正确答案本身就在提供偏好信号！
```

#### GRPO 的优势与局限

| 优势 | 局限 |
|------|------|
| **无 Critic**：省去一半的 RL 显存 | 需要**可验证的奖励**——答案对错必须能自动判断 |
| 组内相对比较天然解决奖励尺度问题 | 对无法客观验证的任务（如写作质量）不适用 |
| 训练稳定（组内标准化消除极端值影响） | 需要一组回复（G 个），增加了采样计算量 |
| 在推理任务上效果突出（DeepSeek-R1 的核心方法） | G 的选择是超参数——太小则 baseline 不准确，太大则计算成本高 |

#### GRPO 与 RLOO 和 REINFORCE 的关系

**RLOO（REINFORCE Leave-One-Out）** 是另一种组内比较方法：

$$\hat{A}_i = r_i - \frac{1}{G-1} \sum_{j \neq i} r_j$$

RLOO 与 GRPO 的区别在于：RLOO 仅减去组内均值（leave-one-out），不做除以标准差的标准化。GRPO 的标准化（z-score）使得 advantage 在不同 batch 间尺度一致，训练更稳定。

**REINFORCE** 是最简单的策略梯度方法：

$$\nabla_\theta J = \mathbb{E}[\nabla_\theta \log \pi_\theta(y|x) \cdot (r - b)]$$

其中 $b$ 是外部 baseline（如移动平均、learned baseline）。与 GRPO 相比，REINFORCE 的 baseline 不随当前组变化，可能引入历史偏差。GRPO 的组内 baseline 是"即时"的，不依赖历史数据，偏差更小。此外，REINFORCE 没有 clip 约束，更新步长不可控，容易导致策略崩溃。

### 7.3 DeepSeek-R1：GRPO 在大规模推理训练中的应用

DeepSeek-R1 是 2025 年初发布的里程碑式模型，展示了**纯 RL 可以显著提升推理能力**：

```
训练流程：

阶段 1: DeepSeek-R1-Zero（纯 RL，无 SFT）
  ├── 基座模型：DeepSeek-V3-Base
  ├── 方法：GRPO
  ├── 奖励设计：
  │   ├── 准确性奖励：最终答案是否正确（数学题、编程题）
  │   └── 格式奖励：是否将推理过程放在 <think>...</think> 标签中
  └── 结果：模型的推理能力显著提升
       ├── AIME 2024 (数学竞赛): 从 15.6% → 71.0% (pass@1)
       └── 但存在一些问题：语言混杂（中英混合）、可读性差

阶段 2: DeepSeek-R1（冷启动数据 + RL）
  ├── 用少量高质量推理链数据（数千条）做冷启动 SFT
  ├── 然后做 GRPO（面向推理的 RL）
  ├── 再收集新数据做 SFT（含推理 + 通用数据）
  └── 最后做全场景 RL（推理 + helpfulness + harmlessness）
```

**DeepSeek-R1 的核心启示**：

1. **纯 RL 就能涌现推理能力**：不需要大量推理数据，只需要可验证的奖励信号
2. **"Aha Moment"（顿悟时刻）**：训练中模型自发学会了"等等，让我重新检查一下"、"换个思路"等反思行为，这些没有被显式编程，而是 RL 优化的涌现结果
3. **格式奖励引导结构化思考**：要求模型将推理过程放入 `<think>` 标签，这种简单的格式约束就足以引导模型进行更深入的思考
4. **语言混杂的自然出现**：模型在推理时会自动在中英文之间切换（选择最"高效"的语言思考），这暗示了推理的语言独立性

### 7.4 GRPO vs PPO vs DPO 对比

| 维度 | PPO (RLHF) | DPO | GRPO |
|------|-----------|-----|------|
| **奖励来源** | Reward Model（人类偏好） | 隐式（偏好对比较） | 可验证奖励（答案对错） |
| **Critic 网络** | ✅ 需要 | ❌ 不需要 | ❌ 不需要 |
| **参考模型** | ✅ 需要（SFT） | ✅ 需要 | ✅ 需要 |
| **训练稳定性** | 低 | 高 | 中-高 |
| **主要优化维度** | Helpfulness, Harmlessness | Helpfulness, Harmlessness | **Reasoning** |
| **是否需要偏好标注** | ✅ 人类偏好 | ✅ 偏好对 | ❌ 自动验证 |
| **典型应用** | ChatGPT, Claude | 开源模型对齐 | DeepSeek-R1, 数学/代码模型 |
| **Online/Offline** | Online | Offline | Online（采样 G 个回复） |

---

## 8. Alignment 方法在各维度上的效果对比

### 8.1 各维度最适用的方法

| Alignment 维度 | 最有效的方法 | 原因 |
|---------------|-------------|------|
| **Helpfulness（有帮助性）** | RLHF (PPO)、DPO | 需要人类偏好数据来定义"多好才算好" |
| **Harmlessness（无害性）** | RLHF、Constitutional AI | 需要明确的拒绝边界；宪法 AI 提供了可审计的价值观框架 |
| **Honesty（诚实性）** | RLHF + 特定数据 | 需要在偏好数据中加入"承认不知道"的范例 |
| **Reasoning（推理能力）** | **GRPO** | 可验证奖励提供了精确的优化信号；RL 天然适合探索推理路径 |
| **Instruction Following（指令遵循）** | SFT + DPO | SFT 打底，DPO 精细化 |

### 8.2 方法组合的实际应用

在实践中，最强的模型通常**组合多种方法**：

**Claude 的训练管线**（Anthropic，大致公开信息）：
```
Pre-training → SFT → Constitutional AI (RLAIF) → RLHF (安全专项)
```

**ChatGPT 的训练管线**（OpenAI，大致公开信息）：
```
Pre-training → SFT → RLHF (PPO) → 安全专项 RLHF
```

**DeepSeek-R1 的训练管线**：
```
Pre-training → 冷启动 SFT (推理数据) → GRPO (推理专项) → SFT (通用能力) → 全场景 RL
```

### 8.3 关键取舍

| 取舍 | 说明 |
|------|------|
| **Helpfulness vs Harmlessness** | 过度追求无害会让模型变得"过度谨慎"——对正常请求也拒绝。需要在训练数据中精细平衡 |
| **Reasoning vs General Helpfulness** | GRPO 增强推理的代价是语言可能变得"机械"（更少的礼貌用语，更直接的推理输出）。需要后续 SFT 恢复 |
| **Alignment Tax（对齐税）** | RLHF/DPO 有时会小幅降低模型在一些学术 benchmark 上的分数（典型数值：MMLU 下降 1-2 个百分点，GSM8K 下降 2-5 个百分点），因为模型学会了拒绝和"承认不知道"而非"硬答"。这是 alignment 的正常代价，需要通过数据质量控制和多阶段训练来最小化 |
| **在线 vs 离线** | 在线方法（PPO, GRPO）能在训练中探索新行为，但成本高；离线方法（DPO）成本低，但受限于已有数据 |

### 8.4 对齐效果的评估基准

| 基准 | 测量维度 | 说明 |
|------|---------|------|
| **MT-Bench** | 多轮对话质量 + 多维度评分 | GPT-4 评分的标准测试，覆盖帮助性、准确性、相关性 |
| **AlpacaEval 2.0** | 指令遵循，对比胜率 | 基于 GPT-4 Turbo 的成对比较，快速评估对齐方法 |
| **SafeRLHF** | 安全性（拒绝有害请求） | 由 PKU 提出的安全评估基准，包含攻击性 prompt 集合 |
| **HarmBench** | 有害内容拒绝率 | 大规模对抗性测试集，覆盖多种攻击模式 |
| **TruthfulQA** | 诚实性（不编造事实） | 测量模型是否重复人类常见误解 |
| **GSM8K / MATH** | 推理能力 | 数学推理基准，特别适合 GRPO 等面向推理的对齐方法评估 |
| **MMLU** | 通用知识广度 | 测量对齐后模型的通用能力保持程度（对齐税检测） |
| **IFEval** | 指令严格遵循 | 精确测量模型是否遵循格式、长度、风格等约束 |

### 8.5 多轮对齐的挑战与方法

当前笔记主要基于**单轮偏好**（prompt → response）的对齐方法。但在实际对话系统中，对齐需要在多轮上下文中进行评估和优化。

**核心挑战**：

1. **对话级 Reward**：最终的对话质量取决于整个对话历史，而非单个回复。需要对整个对话轨迹进行打分，而非孤立评估每一轮
2. **时序信用分配（Temporal Credit Assignment）**：哪个轮次的回复导致了用户的最终偏好？例如，如果用户在第 5 轮感到不满，可能是因为第 2 轮的回答不够准确。标准的单轮 RM 无法处理这种情况
3. **对话一致性**：模型需要在多轮中保持一致的角色、知识立场和价值观（如"上一轮说过的结论不能在下一轮违背"）

**现有解决方案**：

| 方法 | 说明 | 实现难度 |
|------|------|---------|
| **对话级 RM 训练** | 对整个对话历史输入 RM，输出单轮或多轮分数 | 高（需要对话级偏好数据）|
| **Iterative DPO** | 在线收集偏好数据后多次更新模型——每次用当前模型生成新回复，收集偏好，做 DPO | 中（需要重复数据收集流程）|
| **单轮近似** | 将多轮对话拆解为多个独立的单轮偏好对（每轮单独优化） | 低（当前大多数项目的做法）|
| **对话回报分解（GAE）** | 用 GAE（Generalized Advantage Estimation）将对话级的奖励分解到每一轮 | 高（需要 PPO 级别的 RL 基础设施）|

> 💡 **实践经验**：大多数开源项目（如 LLaMA-3、Qwen）目前仍使用**单轮偏好近似**来对齐对话模型。做法是将多轮对话的最后几轮（最关键的回复）提取出来作为单轮偏好对。这个做法虽然不完美，但在实际应用中已被证明足够有效。

### 8.6 各方法显存与速度的量化对比

| 方法 | 训练时模型数量 | 7B 显存估算（不含激活值） | 训练速度（相对 SFT） |
|------|--------------|------------------------|---------------------|
| **SFT** | 1（$\pi_\theta$） | ~14 GB | 1×（基准） |
| **DPO** | 2（$\pi_\theta$, $\pi_{ref}$） | ~28 GB | ~1.2× |
| **ORPO / SimPO** | 1（仅 $\pi_\theta$） | ~14 GB | ~1.1× |
| **RLHF (PPO)** | 4（Actor, Critic, RM, Ref） | ~56 GB | ~3× |
| **GRPO** | 3（Actor, Ref, 无 Critic） | ~42 GB | ~2× |
| **Constitutional AI** | 1-4（取决于阶段） | ~14-56 GB | 视阶段而定 |

> ⚠️ **注意**：以上为模型参数本身的显存估算（BF16/FP16，不含激活值和优化器状态）。加上激活值和 Adam 优化器状态后，实际显存需求约为表中数字的 2-3 倍。例如 DPO 训练 7B 模型实际需要约 50-60 GB 显存。

---

## 9. 总结：Alignment 方法的选型指南

### 9.1 按目标选择

```
你想优化什么？
├── Helpfulness + Harmlessness（通用对齐）
│   ├── 有大量人类偏好数据 → RLHF (PPO) 或 DPO
│   ├── 有有限人类偏好数据 → DPO（离线，成本低）
│   ├── 没有人类偏好数据但有强大 AI → RLAIF
│   └── 需要可审计的价值观 → Constitutional AI
│
├── Reasoning（推理能力）
│   ├── 有可验证奖励（数学/代码） → GRPO
│   ├── 没有可验证奖励 → 用 RLHF/DPO 在推理偏好数据上
│   └── 两者都需要 → 先 GRPO（推理），再 DPO（通用对齐）
│
└── 显存/预算极度有限
    ├── DPO（2 个模型，简单）
    ├── ORPO 或 SimPO（1 个模型，更省）
    └── RLAIF + DPO（标注成本也省了）
```

### 9.2 一句话总结

| 方法 | 一句话 |
|------|--------|
| **SFT** | 教会模型"回答问题"——对齐的起点，不是终点 |
| **RLHF (PPO)** | 用人类偏好训练裁判（RM），再用裁判训练选手（PPO）——经典的"胡萝卜加大棒" |
| **DPO** | 省掉裁判，直接从"哪个更好"的对比中学习——更简单，但缺乏探索 |
| **RLAIF** | 让 AI 当裁判——成本低、大规模，但裁判自身的偏差是隐患 |
| **Constitutional AI** | 给 AI 一本"宪法"，让它对照原则自我修正——可解释、可审计 |
| **GRPO** | 答案对不对自己验算——用客观奖励替代人类偏好，专攻推理 |
| **ORPO / SimPO** | DPO 的轻量化变体——连参考模型都省了，更省显存 |

### 9.3 发展趋势

1. **从复杂到简化**：RLHF（4 模型）→ DPO（2 模型）→ ORPO/SimPO（1 模型）。Alignment 的门槛在不断降低
2. **从人类到 AI**：RLHF（人类偏好）→ RLAIF（AI 偏好）。标注的瓶颈正在被 AI 自身解决
3. **从通用到专业化**：出现了专门针对推理（GRPO）、安全性（Constitutional AI）等特定维度的专用方法
4. **从单阶段到多阶段组合**：最佳实践不再是"用一个方法做所有事"，而是 SFT → 推理 RL → 通用对齐 → 安全对齐 的多阶段管线
5. **可验证奖励的崛起**：对于有客观答案的任务，可验证奖励比人类偏好更精确、更便宜、更易扩展。GRPO 的成功标志着这一范式的重要突破

---

## 10. 工具代码示例

### 10.1 DPOTrainer 代码示例

使用 HuggingFace TRL 库实现 DPO 训练（与 Fine-tuning 笔记中的 LoRA 示例风格一致）：

```python
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from datasets import load_dataset

# 加载模型（需要同时加载参考模型）
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
ref_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

# LoRA 配置
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)

# DPO 配置
dpo_args = DPOConfig(
    output_dir="./dpo_output",
    per_device_train_batch_size=4,
    learning_rate=5e-6,
    beta=0.1,                 # KL 惩罚系数（核心超参数）
    max_length=1024,
    max_prompt_length=512,
    num_train_epochs=1,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
)

# 数据格式：每条需包含 prompt, chosen, rejected 三个字段
dataset = load_dataset("json", data_files="preference_data.jsonl")["train"]

# DPO 训练
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=dpo_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,
)
dpo_trainer.train()
```

### 10.2 PPOTrainer 代码示例

PPO 训练需要更多基础设施（Actor + Critic + RM）：

```python
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
from datasets import load_dataset

# 加载带有 Value Head 的 Actor（自动在 backbone 上加标量输出头）
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

# 参考模型（冻结，用于 KL 惩罚）
ref_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.bfloat16,
)
# Reward Model（需单独训练，这里从已训练好的加载）
reward_model = AutoModelForSequenceClassification.from_pretrained(
    "./my_reward_model",
    torch_dtype=torch.bfloat16,
)

# PPO 配置
ppo_config = PPOConfig(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    learning_rate=1.41e-5,
    batch_size=16,
    mini_batch_size=4,
    gradient_accumulation_steps=1,
    ppo_epochs=4,
    kl_penalty="kl",
    adap_kl_ctrl=True,
    init_kl_coef=0.05,
)

# PPO 训练
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    dataset=load_dataset("json", data_files="prompts.jsonl")["train"],
)

# 训练循环
for epoch in range(3):
    for batch in ppo_trainer.dataloader:
        queries = batch["query"]
        # 1. 生成回复（当前策略）
        responses = ppo_trainer.generate(
            queries,
            pad_token_id=tokenizer.eos_token_id,
            **{"top_p": 0.9, "temperature": 0.7, "max_new_tokens": 256},
        )
        # 2. 计算奖励
        rewards = reward_model(queries, responses)
        # 3. PPO 更新（Actor + Critic）
        stats = ppo_trainer.step(queries, responses, rewards)
```

---

## 参考资源

### 核心论文

- 📄 **InstructGPT / RLHF**: [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — Ouyang et al., 2022
- 📄 **DPO**: [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) — Rafailov et al., 2023
- 📄 **RLAIF**: [RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback](https://arxiv.org/abs/2309.00267) — Lee et al., 2023
- 📄 **Constitutional AI**: [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — Bai et al., 2022
- 📄 **DeepSeek-R1 / GRPO**: [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) — DeepSeek-AI, 2025
- 📄 **ORPO**: [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691) — Hong et al., 2024
- 📄 **SimPO**: [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734) — Meng et al., 2024
- 📄 **LIMA**: [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) — Zhou et al., 2023
- 📄 **Anthropic HHH**: [A General Language Assistant as a Laboratory for Alignment](https://arxiv.org/abs/2112.00861) — Askell et al., 2021

### 工具与代码

- 🏋️ **HuggingFace TRL**: [https://github.com/huggingface/trl](https://github.com/huggingface/trl) — DPO / PPO / SFT 的标准实现
- 🧠 **OpenRLHF**: [https://github.com/OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) — RLHF 分布式训练框架
- 🐳 **veRL**: [https://github.com/volcengine/verl](https://github.com/volcengine/verl) — 字节跳动的 RLHF 框架
- 🏭 **LLaMA-Factory**: [https://github.com/hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) — 支持 DPO/ORPO/SimPO 的一站式微调框架
- 🔬 **DeepSeek-Open-Source**: [https://github.com/deepseek-ai](https://github.com/deepseek-ai) — DeepSeek-R1 及相关工作