# Fine-tuning 技术笔记

---

## 1. 什么是 Fine-tuning？

### 1.1 定义

**Fine-tuning（微调）** 是指在预训练模型（Pre-trained Model）的基础上，使用特定任务或领域的标注数据，对模型参数进行继续训练，使其适应下游任务（Downstream Task）的过程。

这一概念最早可追溯至深度学习在计算机视觉领域的实践：在 ImageNet 上预训练的 ResNet 或 VGG 网络，被广泛微调用于目标检测、语义分割等下游视觉任务。在 NLP 领域，BERT（2018）和 GPT（2018）的出现将"预训练 + 微调"范式推向了主流。

### 1.2 三个核心概念

- **预训练模型（Pre-trained Model）**：在大规模通用语料（如 CommonCrawl、Wikipedia、书籍、GitHub 代码仓库等）上训练得到的基座模型，学习了语言的通用知识、推理能力和统计规律。预训练的目标函数通常为**自回归语言建模**（Autoregressive LM, GPT 系列）或**掩码语言建模**（Masked LM, BERT 系列）。

- **微调数据（Fine-tuning Data）**：针对特定任务的高质量标注数据（例如：指令-回复对、分类标签、问答对、多轮对话等）。数据的质量和多样性直接决定了微调效果的上限。

- **核心思想：迁移学习（Transfer Learning）**：将预训练阶段学到的通用语言知识迁移到特定任务上，避免从零开始训练，大幅降低数据和算力需求。形式化地说，迁移学习利用源域 $\mathcal{D}_S$ 上学到的分布 $P(Y|X)$ 来帮助目标域 $\mathcal{D}_T$ 上的学习任务。

> 💡 **类比1**：预训练模型就像一个通识教育毕业的大学生，微调就是让他接受某个专业（法律、医学、编程等）的岗前培训——基础能力（语言、逻辑）不需要重新培养，只需要补充专业知识和工作规范。

> 💡 **类比2（更技术化）**：可以把预训练权重看作高维参数空间中的一个"好的起点"。从随机初始化训练需要从零探索整个参数空间，而微调只需要在这个好的起点附近做局部优化。这也是为什么微调所需的数据量和算力远小于预训练。

### 1.3 Fine-tuning 的数学形式

设预训练模型的参数为 $\theta_0$（通常已是 $\arg\min_{\theta} \mathcal{L}_{PT}(\theta)$ 的近似解），微调的目标是：

$$\theta^* = \arg\min_{\theta} \mathcal{L}_{FT}(\theta; \mathcal{D}_{FT}), \quad \text{初始化为 } \theta_0$$

其中 $\mathcal{L}_{FT}$ 为下游任务的损失函数，$\mathcal{D}_{FT}$ 为微调数据集。与从随机初始化 $\theta \sim \mathcal{N}(0, \sigma^2)$ 开始训练不同，$\theta_0$ 已经编码了丰富的语言先验，因此优化过程收敛更快、所需数据更少。

对于分类任务（如 BERT 微调做情感分析），通常在预训练模型的最后一层隐藏状态 $\mathbf{h}_{[CLS]}$ 上添加一个线性分类头：

$$P(y | x) = \text{softmax}(W_{cls} \cdot \mathbf{h}_{[CLS]} + b_{cls})$$

其中 $W_{cls} \in \mathbb{R}^{K \times d}$ 为分类权重矩阵，$K$ 为类别数（如情感分析中 $K=2$ 代表正/负），$d$ 为隐藏维度。

对于生成任务（如 GPT 微调做对话），训练目标与预训练一致，仍为 Next Token Prediction，但数据分布从通用文本变为指令-回复对：

$$\mathcal{L}_{SFT} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T_i} \log P_\theta(y_{i,t} | x_i, y_{i,<t})$$

其中 $(x_i, y_i)$ 为指令-回复对，模型仅在 $y_i$ 部分计算损失（通常将 $x_i$ 部分的 loss mask 掉）。

---

## 2. 为什么大语言模型需要 Fine-tuning？

### 2.1 预训练模型的本质局限

预训练的大语言模型（LLM）虽然具备强大的语言生成和推理能力，但其训练目标决定了它与用户期望之间存在根本性的**行为鸿沟（Behavior Gap）**：

| 局限性 | 根本原因 | 具体表现 |
|--------|---------|---------|
| **不会对话** | 预训练目标为无条件/弱条件文本续写，缺少对话格式的交互信号 | 用户问"今天天气怎么样？"，模型可能继续写"她站在窗前，望着灰蒙蒙的天空..." |
| **不会遵循指令** | 预训练语料中极少包含"请帮我..."这类指令性文本 | 无法理解"请用三句话总结以下内容"的任务要求 |
| **领域知识不足** | 通用语料的长尾分布无法覆盖垂直领域 | 医学术语（如"血管紧张素转化酶抑制剂"）、法律条文、芯片设计等专业语境缺失 |
| **输出风格不可控** | 预训练不带格式约束信号 | 无法约束输出为 JSON、YAML、Markdown 或特定模板 |
| **安全对齐缺失** | 基座模型未经过人类偏好优化 | 可能生成有害、偏见或不符合伦理规范的内容（如歧视性言论、危险建议） |

### 2.2 预训练与微调之间的分布偏移

从概率角度看，预训练学习的是通用文本分布 $P_{PT}(text)$，而用户期望的是条件分布 $P_{target}(response | instruction)$。两者之间存在**分布偏移（Distribution Shift）**：

$$P_{PT}(\text{next token} | \text{prefix}) \neq P_{target}(\text{response} | \text{instruction})$$

微调的本质就是通过标注数据来**桥接这个分布偏移**，将模型的行为分布校准到目标分布。

### 2.3 BERT 微调：一个经典案例

以 BERT 微调做情感分析为例，这是理解微调最直观的切入点：

- **预训练阶段**：BERT 在 BooksCorpus（8亿词）和 English Wikipedia（25亿词）上做 MLM + NSP 预训练，学会了理解语言的深层语义和句法结构。此时的 BERT 可以判断"这个词填在这里是否合理"，但不知道"这句话是正面还是负面情感"。
- **微调阶段**：在 IMDb 影评数据集（25K 条标注数据，正面/负面）上微调。BERT 的 Transformer 主体已经具备语义理解能力，只需在 `[CLS]` token 的表示上添加一个二分类头，用交叉熵损失进行训练：

$$\mathcal{L}_{cls} = -\frac{1}{N}\sum_{i=1}^{N} \left[ y_i \log \hat{y}_i + (1-y_i) \log(1-\hat{y}_i) \right]$$

$$\hat{y}_i = \sigma(W_{cls}^T \cdot \mathbf{h}_{[CLS]}^{(i)})$$

- **结果**：微调后的 BERT 在 IMDb 上达到 ~95% 准确率，而从头训练同样的架构在 IMDb 上只能达到 ~88%（因数据量不足以学习语义）。这就是迁移学习的价值——用预训练的通用语义理解能力弥补下游标注数据不足的问题。

> 这个案例揭示了微调的本质辩证法：预训练提供了"理解语言"的能力（what），微调赋予了"完成特定任务"的能力（how）。二者结合，才能将通用智能转化为专用能力。

---

## 3. Pre-training、Fine-tuning、Instruction Tuning 的区别

### 3.1 概念辨析

这三个概念经常被混淆，但它们处于不同的抽象层次：

| 维度 | Pre-training（预训练） | Fine-tuning（微调） | Instruction Tuning（指令微调） |
|------|----------------------|--------------------|------------------------------|
| **数据** | 海量无标注文本（TB 级） | 任务特定的标注数据（数百~数十万条） | 多样化指令-回复对（数万~数百万条） |
| **数据来源** | 网页爬取、书籍、代码仓库 | 人工标注或模型蒸馏 | 人工编写 + 模型蒸馏（Self-Instruct） |
| **训练目标** | 语言建模（Next Token Prediction / MLM） | 适应特定下游任务（分类、生成、QA 等） | 学会理解和遵循人类指令 |
| **输出风格** | 自由文本续写 | 取决于任务（标签、翻译、摘要等） | 有帮助的、安全的对话式回复 |
| **泛化能力** | 通用语言能力（跨任务、跨领域） | 仅限特定任务（窄泛化） | 多任务指令泛化（宽泛化） |
| **典型例子** | LLaMA、GPT-3、Qwen、BERT 基座模型 | BERT → 情感分类器；GPT-2 → 文本摘要 | ChatGPT、Alpaca、Vicuna、Llama-3-Chat |
| **训练范式** | 自监督学习（Self-supervised） | 监督学习（Supervised） | 监督学习（Supervised） |

> 📌 **关键区分**：Instruction Tuning 是 Fine-tuning 的一个子类——所有指令微调都是微调，但不是所有微调都是指令微调。Instruction Tuning 的特异性在于：(1) 数据格式为指令-回复对，(2) 目标是提升跨任务的指令泛化能力，(3) 通常是通向对话模型的必经之路。

### 3.2 完整训练管线

```
原始文本语料（数 TB）
    │
    ▼
┌────────────────────────────┐
│  Pre-training              │
│  目标：Next Token Prediction│
│  数据：无标注文本            │
│  产出：Base Model（基座模型） │
└──────────┬─────────────────┘
           │
           ├──→ Fine-tuning（任务特定微调）
           │    目标：下游任务 Loss
           │    数据：标注样本（数百~数千）
           │    产出：Task-specific Model
           │    示例：BERT → 情感分析 / NER / 文本分类
           │
           └──→ Instruction Tuning（指令微调）
                 目标：Next Token Prediction（但数据分布变为指令-回复）
                 数据：指令-回复对（数万~数百万）
                 产出：Chat Model / Instruct Model
                       │
                       ├──→ RLHF（PPO + Reward Model）
                       │    进一步对齐人类偏好
                       │
                       └──→ DPO（Direct Preference Optimization）
                             无需奖励模型的偏好对齐
```

> 🔗 **与《Alignment 笔记》的关联**：图中的 Instruction Tuning（SFT）既是**微调（Fine-tuning）的一种形式**，也是**对齐（Alignment）的起点**。SFT 教会模型"回答问题"，而 RLHF/DPO 等对齐方法在 SFT 的基础上进一步优化模型的行为。详细的对齐方法体系（DPO、RLHF、GRPO、RLAIF 等）见《Alignment 技术笔记》。

### 3.3 预训练目标函数的数学对比

不同预训练范式的目标函数决定了模型的行为倾向：

**自回归语言模型（GPT 系列）**：

$$\mathcal{L}_{AR} = -\sum_{t=1}^{T} \log P_\theta(x_t | x_{<t})$$

模型按从左到右顺序预测下一个 token，天然适合生成任务。但只能看到左侧上下文（单向）。

**掩码语言模型（BERT 系列）**：

$$\mathcal{L}_{MLM} = -\sum_{i \in \mathcal{M}} \log P_\theta(x_i | x_{\backslash \mathcal{M}})$$

其中 $\mathcal{M}$ 为被随机掩码（15%）的 token 位置集合。模型同时利用左右两侧上下文预测被掩码的词，天然适合理解任务。但无法直接用于自回归生成。

**前缀语言模型（T5、GLM）**：

$$\mathcal{L}_{Prefix} = -\sum_{t=|\mathbf{p}|+1}^{|\mathbf{p}|+|\mathbf{t}|} \log P_\theta(x_t | \mathbf{p}, x_{|\mathbf{p}|+1:t-1})$$

结合了双向编码（前缀部分）和自回归生成（目标部分）的优势。

---

## 4. Full Fine-tuning vs Parameter-Efficient Fine-tuning

### 4.1 Full Fine-tuning（全量微调）

#### 定义与特点

- **定义**：更新模型的**所有参数** $\theta$（通常数亿 ~ 数千亿）。
- **特点**：参数自由度最大，理论上模型适应能力最强。但显存和存储成本极高，且存在灾难性遗忘风险。

#### 显存消耗的精确分析

训练时的显存开销主要由四部分组成：

$$\text{Total Memory} \approx \underbrace{\text{Model Params}}_{P} + \underbrace{\text{Gradients}}_{P} + \underbrace{\text{Optimizer States}}_{2P \text{ (Adam)}} + \underbrace{\text{Activations}}_{\text{取决于 batch size 和序列长度}}$$

对于 Adam 优化器，优化器状态包含 $m_t$（一阶矩估计）和 $v_t$（二阶矩估计），各需一份 FP32 存储。详细分解：

| 组件 | 精度 | 大小（7B 模型） | 说明 |
|------|------|---------------|------|
| 模型参数 $\theta$ | FP16/BF16 | ~14 GB | 前向和反向传播 |
| 梯度 $\nabla_\theta \mathcal{L}$ | FP16/BF16 | ~14 GB | 反向传播产生 |
| 优化器状态 $m_t$ | FP32 | ~28 GB | Adam 一阶矩（FP32 主副本） |
| 优化器状态 $v_t$ | FP32 | ~28 GB | Adam 二阶矩（FP32 主副本） |
| 激活值 | FP16 | ~28-56 GB | 取决于 batch size 和序列长度 |
| **合计（不含激活值）** | | **~84 GB** | $14 + 14 + 28 + 28$ |
| **合计（含激活值）** | | **~112-140 GB** | |

> 这就是 $\text{模型参数} \times 16$ (bytes) 经验公式的来源：2 bytes (FP16 参数) + 2 bytes (FP16 梯度) + 4 bytes (FP32 $m$) + 4 bytes (FP32 $v$) + 4 bytes (FP32 参数主副本用于更新) = 16 bytes/参数。即 $7\text{B} \times 16 \text{ bytes} = 112 \text{ GB}$（不含激活值）。

#### Adam 优化器的数学形式（理解显存来源）

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad \text{(一阶矩，FP32)}$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad \text{(二阶矩，FP32)}$$

$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} \quad \text{(参数更新)}$$

这三个式子说明了为什么 Adam 需要 3 份额外的参数存储（$m_t$, $v_t$, master weights）。

### 4.2 Parameter-Efficient Fine-tuning（PEFT，参数高效微调）

#### 定义

**PEFT** 的核心思想是冻结绝大多数预训练参数，仅优化极少量的额外参数 $\phi$（通常 $|\phi| \ll |\theta|$）：

$$\theta_{FT} = \theta_0 \cup \phi^*, \quad \phi^* = \arg\min_{\phi} \mathcal{L}(\theta_0, \phi; \mathcal{D}_{FT})$$

其中 $\theta_0$ 冻结不变，$\phi$ 为可训练的新参数，$|\phi|$ 通常仅占 $|\theta_0|$ 的 0.01% ~ 5%。

#### 四种主流 PEFT 方法深度对比

| 维度 | Adapter | Prefix Tuning | Prompt Tuning | LoRA |
|------|---------|---------------|---------------|------|
| **提出时间** | 2019 (Houlsby et al.) | 2021 (Li & Liang) | 2021 (Lester et al.) | 2021 (Hu et al.) |
| **核心思想** | 在 Transformer 层间插入小型瓶颈网络 | 在每层的 Key/Value 前添加可训练的连续前缀 | 在输入 embedding 前添加可学习的 soft prompt | 用低秩矩阵 $BA$ 近似权重更新 |
| **插入位置** | 每层 Attention 和 FFN 之后 | 每层 Multi-Head Attention 的 K/V | 仅输入层 | 每层的 Attention 投影矩阵 |
| **可训练参数** | ~2%–5% | ~0.1%–1% | ~0.001%–0.01% | ~0.1%–1% |
| **推理额外延迟** | 有（串行瓶颈层） | 有（KV cache 增长） | 无（仅改输入） | **无（可合并回权重）** |
| **效果（小模型）** | 好 | 中等 | 较差 | 好 |
| **效果（大模型 > 10B）** | 好 | 好 | 好 | 好 |
| **多任务切换** | 换 Adapter 模块 | 换前缀向量 | 换 prompt 向量 | 换 LoRA 权重 |
| **存储（每任务）** | ~几十 MB | ~几 MB | ~几 KB | ~几 MB |

#### Adapter 详解

Adapter 在每个 Transformer 子层后插入一个**瓶颈网络（Bottleneck）**：

$$\mathbf{h}_{out} = \mathbf{h}_{in} + f_{act}(\mathbf{h}_{in} \cdot W_{down}) \cdot W_{up}$$

其中 $W_{down} \in \mathbb{R}^{d \times d_{bottle}}$（降维），$W_{up} \in \mathbb{R}^{d_{bottle} \times d}$（升维），$d_{bottle} \ll d$（如 64 vs 4096），$f_{act}$ 为非线性激活函数。

- **优点**：简单直接，可插拔；每个 Adapter 参数量可控
- **缺点**：推理时需要逐层计算 Adapter 的额外前向传播，引入 ~3%-5% 的延迟；无法像 LoRA 一样合并权重消除开销

#### Prefix Tuning 详解

在每一层 Transformer 的 Key 和 Value 矩阵前拼接 $l$ 个可训练的连续前缀向量：

$$\mathbf{K}_{aug} = [\mathbf{P}_K; \mathbf{K}], \quad \mathbf{V}_{aug} = [\mathbf{P}_V; \mathbf{V}]$$

其中 $\mathbf{P}_K, \mathbf{P}_V \in \mathbb{R}^{l \times d}$ 为可学习的前缀参数。

- **优点**：参数极少（仅需 $2 \times l \times d \times N_{layers}$ 个参数）
- **缺点**：占用部分序列长度（有效上下文窗口减少 $l$ 个 token）；推理时 KV cache 更长；对初始化和优化较敏感（通常需要重参数化技巧来稳定训练）

#### Prompt Tuning 详解

仅在**输入层**添加 $l$ 个可学习的 soft prompt embedding：

$$\mathbf{E}_{aug} = [\mathbf{P}; \mathbf{E}(x)]$$

其中 $\mathbf{P} \in \mathbb{R}^{l \times d}$ 为可学习的 prompt embedding，$\mathbf{E}(x)$ 为输入文本的 embedding。

- **优点**：参数数量最少（仅 $l \times d$ 即几千个参数），极致轻量
- **缺点**：对中小规模模型（< 10B）效果较差；收敛慢；效果上限低

> 🔬 **一个有趣的实验结论**：Lester et al. (2021) 发现，对于 10B+ 参数量的模型，Prompt Tuning 的效果可以与 Full Fine-tuning 媲美；但对于 < 1B 参数的模型，Prompt Tuning 的效果远不如全量微调。这暗示了一个规律：**模型越大，对参数扰动越敏感，越小的参数变化就能引起显著的行为改变**。这也是 PEFT 在大模型时代流行的深层原因。

### 4.3 Full Fine-tuning 与 PEFT 的本质对比

从优化的角度看：

- **Full Fine-tuning** 在原始高维参数空间 $\mathbb{R}^{|\theta|}$ 中搜索最优解，搜索空间巨大但灵活性最高。
- **PEFT** 将搜索限制在低维子流形（low-dimensional submanifold）$\{ \theta_0 + \Delta\theta(\phi) \}$ 上，$\dim(\phi) \ll \dim(\theta)$。这是对解空间的**结构约束**，牺牲了部分灵活性，但换来了更好的泛化（类似 Occam's Razor 原理）。

这解释了为什么 PEFT 在小数据集上不容易过拟合：低维搜索空间天然具有正则化效应。

| 对比维度 | Full Fine-tuning | PEFT（以 LoRA 为例） |
|---------|-----------------|---------------------|
| **可训练参数** | 100% | ~0.1% ~ 1% |
| **搜索空间维度** | $|\theta|$ （数十亿~数千亿） | $r \times (d + k)$（数百万） |
| **显存需求** | 极高（~16× 参数量） | 低（约为全量的 1/10 或更低） |
| **存储成本** | 每任务一个完整模型（~14GB ~ 140GB） | 每任务一个轻量适配器（几 MB ~ 几十 MB） |
| **过拟合风险** | 数据少时较高 | 较低（低维约束的正则化效应） |
| **灾难性遗忘风险** | 高（所有参数被修改） | 低（原始权重冻结） |
| **推理延迟** | 正常（无额外计算） | 正常（LoRA 权重可合并，无额外开销） |
| **多任务支持** | 需切换整个模型（重新加载数十 GB） | 热插拔适配器（毫秒级切换） |
| **分布式训练难度** | 高（通信量大） | 低（仅同步少量梯度） |

### 4.4 模型合并（Model Merging）：微调的扩展

模型合并是近年兴起的**无需额外训练即可组合多个模型能力**的技术。给定多个在不同领域微调后的模型 $\{\theta_1, \theta_2, ..., \theta_n\}$（可以是全量权重或 LoRA 适配器），合并得到一个综合能力更强的模型 $\theta_{merged}$：

**常用合并方法**：

| 方法 | 数学形式 | 核心思想 |
|------|---------|---------|
| **Linear Merge** | $\theta_{merged} = \sum_{i=1}^{n} w_i \theta_i$ | 加权平均，$w_i$ 为每模型的权重系数 |
| **TIES-Merging** | 1. 提取每模型的任务向量 $\tau_i = \theta_i - \theta_0$ <br> 2. 裁剪符号冲突的参数 <br> 3. 对剩余参数平均 | 解决不同模型间参数更新的冲突 |
| **DARE** | 1. 随机丢弃大部分任务向量（~90%） <br> 2. 缩放剩余向量 <br> 3. 合并 | 稀疏合并，消除冗余更新 |

**与微调的关系**：

1. **先微调后合并**：可以分别微调多个"专家"LoRA（如编程专家、数学专家、安全专家），然后合并为一个全能 LoRA，无需同时训练
2. **降低存储成本**：存储 N 个专家 → 存储 1 个合并模型
3. **工具支持**：[mergekit](https://github.com/arcee-ai/mergekit) 是目前最主流的模型合并框架，支持 TIES、DARE、Linear 等多种策略

> 💡 **实践建议**：如果需要在多个领域间平衡能力（如同时提升编程和数学），分别微调两个 LoRA 再合并，往往比用混合数据一次性训练效果更好。合并后的模型在各项任务上的表现通常接近甚至超过各自的专家模型。

---

## 5. LoRA（Low-Rank Adaptation）深度解析

### 5.1 理论动机：本征维度假说

LoRA 的设计深受**本征维度（Intrinsic Dimension）**研究的启发。Aghajanyan et al. (2020) 发现：即使模型有数十亿参数，其参数更新的有效自由度（即本征维度）远小于参数总量。对于大多数下游任务，在一个仅几百维的随机子空间中优化就足以达到全参数微调 90% 以上的效果。

这暗示：

$$\text{rank}(\Delta W) \ll \min(d, k)$$

其中 $\Delta W$ 是微调带来的权重变化量。LoRA 正是利用这一洞察，显式地将 $\Delta W$ 建模为两个低秩矩阵的乘积。

### 5.2 矩阵低秩分解的数学基础

任意矩阵 $M \in \mathbb{R}^{d \times k}$ 的 SVD 分解为：

$$M = U \Sigma V^T = \sum_{i=1}^{\min(d,k)} \sigma_i \mathbf{u}_i \mathbf{v}_i^T$$

其中 $\sigma_1 \geq \sigma_2 \geq ... \geq \sigma_{\min(d,k)} \geq 0$ 为奇异值。如果大多数奇异值接近于 0，则可以用前 $r$ 个最大的奇异值（及对应向量）近似 $M$：

$$M \approx \sum_{i=1}^{r} \sigma_i \mathbf{u}_i \mathbf{v}_i^T = \tilde{U} \tilde{\Sigma} \tilde{V}^T$$

这个近似的误差可由 Eckart-Young-Mirsky 定理给出：

$$\min_{\text{rank}(M_r) \leq r} \|M - M_r\|_F = \sqrt{\sum_{i=r+1}^{\min(d,k)} \sigma_i^2}$$

LoRA 的洞见在于：**微调中的权重更新 $\Delta W$ 恰好具有这种低秩性质**，因此可以直接用 $B A$ 来参数化它，其中 $B \in \mathbb{R}^{d \times r}$、$A \in \mathbb{R}^{r \times k}$，且 $r \ll \min(d, k)$。

### 5.3 核心公式推导

**原始前向传播**：

$$h = W_0 x$$

其中 $W_0 \in \mathbb{R}^{d \times k}$ 为预训练权重（冻结），$x \in \mathbb{R}^{k}$ 为输入向量，$h \in \mathbb{R}^{d}$ 为输出向量。

**LoRA 修改后**：

$$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} \cdot BA x$$

其中：

- $B \in \mathbb{R}^{d \times r}$：**可训练的低秩矩阵**（上投影矩阵），通常用高斯分布 $\mathcal{N}(0, \sigma^2)$ 初始化
- $A \in \mathbb{R}^{r \times k}$：**可训练的低秩矩阵**（下投影矩阵），通常用 $\mathcal{N}(0, \sigma^2)$ 初始化
- $r \ll \min(d, k)$：秩（rank），典型值为 4、8、16、32、64
- $\alpha$：缩放因子（Scaling Factor），用于控制 LoRA 更新的幅度
- $\frac{\alpha}{r}$：归一化缩放，使得改变 $r$ 时不需要调整学习率（因为 $BA$ 的元素数量随 $r$ 线性增长，除以 $r$ 抵消了这个效应）

**关键设计选择——$A$ 的初始化**：

$A$ 使用 Kaiming 均匀初始化，而 **$B$ 初始化为零矩阵**。这在训练开始时保证 $\Delta W = 0$，模型的初始行为完全由 $W_0$ 决定，避免了随机噪声影响。随着训练进行，$A$ 首先学习有意义的降维投影，$B$ 随后学习如何将投影后的表示映射回原始空间。

**推理时的权重合并**：

$$W_{\text{merged}} = W_0 + \frac{\alpha}{r} BA$$

合并后的计算量与原始模型完全一致，这是 LoRA 相对于 Adapter 的核心优势之一。合并操作的数学意义是将低秩更新"吸收"进原始权重，使模型在推理时无需维护额外的模块。

### 5.4 参数量计算

对于 Transformer 中的一个线性层 $W \in \mathbb{R}^{d \times k}$（如 $d = k = 4096$）：

- 原始参数量：$d \times k = 4096^2 \approx 16.8\text{M}$
- LoRA 参数量（$r=8$）：$d \times r + r \times k = 4096 \times 8 + 8 \times 4096 = 65,536 \approx 0.066\text{M}$
- 参数量比：$\frac{65,536}{16,777,216} \approx 0.39\%$

对于 LLaMA-7B 模型（总计 ~6.7B 参数，包含 32 层，每层 4 个 Attention 投影矩阵 $q,k,v,o$）：

- 全量参数：~6.7B
- LoRA 参数（$r=8$，仅 $q,v$）：~4.2M（约 0.06%）
- LoRA 参数（$r=8$，$q,k,v,o$）：~8.4M（约 0.12%）

### 5.5 `target_modules`：应该对哪些矩阵应用 LoRA？

LoRA 通常应用在 **Attention 层的线性投影矩阵**上，因为 Attention 是模型中最关键的语义交互层。

对于 LLaMA 架构，其 Attention 计算为：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中 $Q = W_q X$, $K = W_k X$, $V = W_v X$，输出 $O = W_o \cdot \text{Attention}(Q, K, V)$。

**各投影矩阵的作用：**

| 矩阵 | 维度 | 作用 | LoRA 重要性 |
|------|------|------|-----------|
| $W_q$ (q_proj) | $d \times d_{head} \cdot n_{heads}$ | 将输入投影为 Query，决定"我要查什么" | ⭐⭐⭐ 最关键 |
| $W_k$ (k_proj) | $d \times d_{head} \cdot n_{heads}$ | 将输入投影为 Key，决定"我是什么内容" | ⭐⭐ |
| $W_v$ (v_proj) | $d \times d_{head} \cdot n_{heads}$ | 将输入投影为 Value，承载实际信息内容 | ⭐⭐⭐ 最关键 |
| $W_o$ (o_proj) | $d_{head} \cdot n_{heads} \times d$ | 将多头输出合并回原始维度 | ⭐⭐ |
| $W_{gate}$ (gate_proj) | $d \times d_{ff}$ | FFN 的门控（SwiGLU 架构） | ⭐（可选） |
| $W_{up}$ (up_proj) | $d \times d_{ff}$ | FFN 的上投影 | ⭐（可选） |
| $W_{down}$ (down_proj) | $d_{ff} \times d$ | FFN 的下投影 | ⭐（可选） |

**实践经验**：

- **最小配置**：`["q_proj", "v_proj"]` — 性价比最高，覆盖最关键的信息查询和内容承载
- **标准配置**：`["q_proj", "k_proj", "v_proj", "o_proj"]` — 覆盖完整 Attention 通路
- **增强配置**：`["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` — 覆盖 Attention + FFN，效果最好但参数量最大

> 📊 **消融实验**：Hu et al. (2021) 在 GPT-3 175B 上的实验表明，仅在 $q$ 和 $v$ 上应用 LoRA 就能达到全量微调效果的 ~95%；加上 $k$ 和 $o$ 能再提升约 2-3%；把 FFN 也加入后提升有限（< 1%），但参数量翻倍。

### 5.6 为什么 LoRA 有效？从本征维度和流形学习的视角

1. **本征维度低**：下游任务的 $\Delta W$ 确实集中在低秩子空间中，LoRA 通过显式建模这个低秩结构，在不牺牲太多表达能力的条件下大幅减少参数。
2. **隐式正则化**：限制 $\Delta W$ 为低秩等价于对解空间施加了平滑约束，降低了过拟合风险。
3. **初始化友好**：$A$ 随机初始化 + $B$ 零初始化保证训练初期的稳定性，确保微调不会立刻破坏预训练知识。
4. **可合并性**：$W_{merged} = W_0 + \frac{\alpha}{r}BA$ 是线性操作，推理时无需额外计算，这是 LoRA 在工程上最重要的优势。

### 5.7 LoRA 超参数的选择

| 超参数 | 推荐值 | 逻辑 |
|--------|--------|------|
| $r$ (秩) | 4 / 8 / 16 | 数据少（< 1K）用小 $r$；数据多（> 10K）用大 $r$ |
| $\alpha$ (缩放) | $r \times 2$ 或 $r \times 4$ | $\alpha/r$ 决定了 LoRA 分支相对于主分支的权重 |
| dropout | 0.05 ~ 0.1 | 防过拟合；数据越少越高 |
| target_modules | `q,v`（最小） / `q,k,v,o`（标准） | 视任务复杂度而定 |

> ⚠️ **常见误解**：$r$ 越大并不总是越好。当 $r$ 超过数据的"信息维度"时，增加的参数只是拟合噪声。在实践中，$r=8$ 和 $r=64$ 的效果差距通常很小（< 1%）。

### 5.8 代码示例（使用 HuggingFace PEFT 库）

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载基座模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# 配置 LoRA
lora_config = LoraConfig(
    r=8,                                           # 秩（核心超参数）
    lora_alpha=32,                                 # 缩放因子
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 目标模块
    lora_dropout=0.1,                              # LoRA 层的 dropout
    bias="none",                                   # 不训练 bias（减少参数）
    task_type=TaskType.CAUSAL_LM,                  # 任务类型
)

# 注入 LoRA 适配器
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 输出示例：trainable params: 8,388,608 || all params: 6,746,804,224 || trainable%: 0.1243%

# 训练完成后保存
model.save_pretrained("./my_lora_weights")  # 仅保存 ~16MB
```

### 5.9 其他低秩适配变体（2024-2025）

近年来，在 LoRA 的基础上涌现了一系列改进变体，进一步推动了 PEFT 的效果上限和效率边界。

#### DoRA（Weight-Decomposed Low-Rank Adaptation, 2024）

**核心思想**：将预训练权重 $W_0$ 分解为**幅度（magnitude）**和**方向（direction）**两个分量：

$$W_0 = m \cdot \frac{V}{\|V\|}, \quad \Delta W_{\text{DoRA}} = \frac{m + \Delta m}{\|V + \Delta V\|} \cdot (V + \Delta V) - W_0$$

其中 $m \in \mathbb{R}^{1 \times k}$ 为可学习的幅度向量，$\Delta V$ 使用 LoRA 的低秩分解 $\Delta V = BA$ 来参数化。

**与 LoRA 的区别**：

| 维度 | LoRA | DoRA |
|------|------|------|
| 更新方式 | $\Delta W = BA$（同时影响幅度和方向） | 幅度 $m$ 直接学习，方向 $\Delta V$ 用 LoRA |
| 参数约束 | 幅度和方向的更新耦合在一起 | 解耦后方向更新更自由 |
| 效果 | 在复杂任务上可能受幅度约束限制 | 在推理、数学等任务上常优于 LoRA |
| 额外开销 | 无 | 极小（仅多一个可学习幅度向量） |

> **实践建议**：DoRA 在许多基准上（如 GSM8K、MMLU）比 LoRA 提升 1-3%，且不增加推理延迟（可合并回权重）。如果你的任务需要较强的推理能力（数学、编程），值得尝试。

#### AdaLoRA（Adaptive Low-Rank Adaptation, 2023）

**核心思想**：LoRA 对所有层、所有矩阵使用相同的 $r$，但实际不同模块的重要性差异很大。AdaLoRA 根据**重要性分数**动态分配秩：

$$\mathcal{L}_{AdaLoRA} = \mathcal{L}_{FT} + \lambda \sum_{i} \sum_{j} S_{i,j}$$

其中 $S_{i,j}$ 为第 $i$ 个模块的第 $j$ 个奇异值的重要性分数，通过正则化更新自动学习。

- **好处**：重要模块（如深层 Attention）获得更大 $r$，不重要模块（如浅层 FFN）获得更小 $r$，总参数量不变但分配更高效
- **代价**：训练时需维护 SVD 分解和重要性估计，计算开销略高于 LoRA
- **适用场景**：模块间重要度差异大的模型（不同层深度差异大时效果更明显）

#### VeRA（Vector-based Random Matrix Adaptation, 2023）

**核心思想**：共享一对随机初始化且**冻结**的低秩矩阵 $B_{shared}, A_{shared}$，仅学习每层的缩放向量 $\mathbf{b} \in \mathbb{R}^d$ 和 $\mathbf{d} \in \mathbb{R}^k$：

$$\Delta W = \mathbf{b} \odot B_{shared} \cdot A_{shared} \odot \mathbf{d}$$

其中 $\odot$ 表示逐元素广播乘法，$B_{shared} \in \mathbb{R}^{d \times r}$ 和 $A_{shared} \in \mathbb{R}^{r \times k}$ 在所有层共享且固定不变。

| 方法 | 参数量（7B, r=8） | 存储 |
|------|------------------|------|
| LoRA | ~8.4M | ~17 MB |
| VeRA | ~0.8M | ~1.6 MB |

- **极致轻量**：参数量仅为 LoRA 的 **1/10**，适合极端资源受限场景
- **效果损失**：在大多数任务上比同 $r$ 的 LoRA 低 2-5%，但参数量节省显著

#### LoRA+（2024）

**核心思想**：为 LoRA 中的 $A$ 和 $B$ 矩阵设置**不同的学习率**：

$$\eta_B = k \cdot \eta_A, \quad k \approx 2 \text{（推荐值）}$$

- **理论动机**：$A$ 控制特征选择的方向，$B$ 控制映射的幅度。两者在训练中的梯度方差差异很大，统一学习率会导致一方过调或欠调
- **实际效果**：收敛速度提升约 2 倍，训练更稳定
- **兼容性**：可直接替换标准 LoRA 训练中的学习率设置，无需修改模型结构

#### PiSSA（Principal Singular values and Singular vectors Adaptation, 2024）

**核心思想**：对预训练权重 $W_0$ 做 SVD 分解，用**主要奇异值**来初始化 LoRA 的 $A, B$：

$$W_0 = U \Sigma V^T = \underbrace{U_r \Sigma_r^{\frac{1}{2}}}_{B_{init}} \cdot \underbrace{\Sigma_r^{\frac{1}{2}} V_r^T}_{A_{init}} + \underbrace{U_{\backslash r} \Sigma_{\backslash r} V_{\backslash r}^T}_{\text{冻结的残差}}$$

- **与标准 LoRA 的区别**：LoRA 的 $A$ 和 $B$ 从随机初始化开始，而 PiSSA 直接从预训练权重中继承"主方向"
- **效果**：收敛更快，最终效果通常优于同秩 LoRA（尤其在数据量较少的场景）
- **关键细节**：$A_{init}$ 和 $B_{init}$ 从权重的主要奇异值初始化，剩余的 $\backslash r$ 方向的权重被冻结，构成一个新的、更小的"基座模型"

#### 各方法速查

| 方法 | 年份 | 核心思想 | 参数量 vs LoRA | 效果 vs LoRA | 推荐场景 |
|------|------|---------|---------------|-------------|---------|
| **DoRA** | 2024 | 幅度/方向解耦 | ≈ LoRA（略多） | ↑ 1-3% | 推理/数学任务 |
| **AdaLoRA** | 2023 | 自适应秩分配 | ≈ LoRA | ↑ 1-2% | 模块重要度差异大的模型 |
| **VeRA** | 2023 | 共享随机矩阵+缩放向量 | ↓ 90% | ↓ 2-5% | 极端资源受限 |
| **LoRA+** | 2024 | $A,B$ 不同学习率 | = LoRA | → 收敛快 2× | 任何 LoRA 场景（即插即用） |
| **PiSSA** | 2024 | SVD 主方向初始化 | = LoRA | ↑ 1-3% | 少数据场景 |

> 💡 **工程建议**：在日常项目中，**DoRA 和 LoRA+ 是最值得优先尝试的 LoRA 改进方法**。DoRA 效果提升稳定且无推理开销，LoRA+ 仅需改一行学习率配置。在开源项目中（如 LLaMA-Factory、Axolotl），这二者已原生支持。

---

## 6. QLoRA（Quantized LoRA）深度解析

### 6.1 背景：为什么需要量化 + LoRA？

LoRA 虽然大幅降低了可训练参数的数量，但**基座模型本身的参数**仍需以 FP16/BF16 精度加载到显存中。对于 65B 模型，仅加载参数就需要 ~130GB 显存（FP16），单卡无法承载。QLoRA 的目标是**同时降低基座模型和 LoRA 参数的显存需求**，使消费级硬件也能微调大模型。

### 6.2 量化的数学基础

量化（Quantization）将一个浮点数张量 $\mathbf{X} \in \mathbb{R}^{n}$ 映射到一个低精度表示：

$$X_i^{quant} = \text{round}\left(\frac{X_i^{fp} - \text{offset}}{\text{scale}}\right), \quad X_i^{fp} \approx X_i^{quant} \times \text{scale} + \text{offset}$$

对于 $b$-bit 量化，量化后的取值有 $2^b$ 种可能。4-bit 即 16 种取值。

**均匀量化 vs NF4 的信息论对比**：

假设权重 $w \sim \mathcal{N}(0, \sigma^2)$，量化位数 $b = 4$：

- **均匀量化**：将 $[-k\sigma, k\sigma]$ 等分为 16 个区间。但由于正态分布的集中性（~68% 在 $[-\sigma, \sigma]$），两侧的量化区间极少使用，信息损失大。
- **NF4 量化**：将正态分布的累积分布函数（CDF）的 $[0, 1]$ 等分为 16 份，每份对应的分位数作为量化值。这样每个量化值被使用的概率相等（$\frac{1}{16}$），在信息论意义上最优。

**NF4 的数学构造**：

设 $\Phi(\cdot)$ 为标准正态分布的 CDF，$q_i = \Phi^{-1}\left(\frac{i}{16} + \frac{1}{32}\right)$ 为第 $i$ 个量化区间的中点（分位数），则 NF4 的量化值为：

$$\text{NF4}_i = q_i, \quad i = 0, 1, ..., 15$$

这些值被归一化到 $[-1, 1]$ 后用于量化。

### 6.3 QLoRA 的三大核心技术

#### ① 4-bit NormalFloat (NF4)

- **核心洞察**：预训练神经网络的权重经验分布高度近似正态分布 $\mathcal{N}(0, \sigma^2)$。
- **NF4 的优势**：信息论最优——在正态分布假设下，NF4 是使量化误差期望最小的 4-bit 数据类型。
- **与标准 INT4 的对比**：
  - INT4：线性均匀量化，范围 $[-8, 7]$（有符号）
  - NF4：非线性量化，每个量化区间的概率质量相等，对尾部（大权重）有更好的分辨率

#### ② 双重量化 (Double Quantization)

以块量化（block-wise quantization）为例，将权重矩阵分成大小为 $B$ 的块（如 $B=64$），每块有自己的量化常数 $c_j \in \mathbb{R}$（32-bit float）。对于 7B 模型，$B=64$ 时量化常数的存储为：

$$\frac{7 \times 10^9}{64} \times 4 \text{ bytes} = 437.5 \text{ MB}$$

量化常数本身占据了不可忽略的存储。双重量化对这些 $c_j$ 再进行 8-bit 量化：

$$c_j^{dq} = \text{Quantize}_8(c_j; c_{mean})$$

其中 $c_{mean}$ 为 FP32 均值。这样量化常数的存储从 32-bit/块降至 8.5 bit/块（8-bit 量化值 + 共享的 FP32 均值），额外节省约 0.5 bit/参数。

**双重量化的数学形式**：

$$\text{DoubleQuant}(W) = \text{NF4}(W; \text{INT8}(c; c_{FP32}))$$

第一层：权重量化为 NF4（4-bit）  
第二层：量化常数再量化为 INT8（8-bit）  
净效果：~4.0 bit/参数（而非 4.5 bit/参数）

#### ③ 分页优化器 (Paged Optimizer)

- **内存管理策略**：统一内存（Unified Memory）实现 GPU ↔ CPU 的自动页面迁移。当 GPU 显存不足时，将不活跃的优化器状态页面自动 evict 到 CPU RAM；需要时触发 page fault 换回 GPU。
- **与操作系统的类比**：正如虚拟内存让程序以为自己有比物理内存更大的地址空间一样，Paged Optimizer 让训练过程以为自己有比 GPU 显存更大的"优化器内存"。
- **实际意义**：防止因训练过程中的瞬时显存峰值（如处理一个特别长的序列时）导致的 OOM，使训练更加鲁棒。

### 6.4 QLoRA 中 LoRA 的特殊处理

QLoRA 的一个关键细节是：LoRA 适配器的参数以 **BF16** 精度存储和训练，不经过量化。这是因为：

1. LoRA 参数极少（几 MB），BF16 不会造成显著的显存压力
2. 如果对 LoRA 参数也做量化，会引入额外的近似误差，削弱微调的能力
3. 前向传播时，BF16 的 LoRA 输出与 NF4 反量化后的基座模型输出相加即可

前向传播流程：

$$h = \text{dequant}(W_0^{NF4}) \cdot x^{BF16} + \frac{\alpha}{r} (BA)^{BF16} \cdot x^{BF16}$$

其中 $\text{dequant}(\cdot)$ 将 NF4 权重反量化为 BF16 参与计算，计算完成后结果仍为 BF16。

### 6.5 显存对比（扩展版）

| 方法 | 7B | 13B | 33B | 65B | 70B | 可运行硬件（最低） |
|------|-----|------|------|------|------|-----------------|
| Full Fine-tuning (FP32) | ~112 GB | ~208 GB | ~528 GB | ~1 TB | ~1.04 TB | 4-8×A100 80GB |
| Full Fine-tuning (混合精度) | ~84 GB | ~156 GB | ~396 GB | ~780 GB | ~840 GB | 2-4×A100 80GB |
| LoRA (FP16, r=8) | ~16 GB | ~30 GB | ~74 GB | ~140 GB | ~148 GB | 1×A100 40GB |
| LoRA (INT8, r=8) | ~10 GB | ~19 GB | ~46 GB | ~86 GB | ~92 GB | 1×A6000 48GB |
| **QLoRA (NF4, r=8)** | **~6 GB** | **~10 GB** | **~20 GB** | **~33 GB** | **~38 GB** | 1×RTX 3090/4090 24GB |

> 📊 从表中可以看出：QLoRA 将 65B 模型的微调成本从 ~1TB（Full FT）降至 ~33GB，降幅约 **30 倍**。这个数量级的降低使得个人开发者和学术研究团队也能处理百亿级模型。

### 6.6 QLoRA 代码示例

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# ---------- 4-bit 量化配置 ----------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                          # 以 4-bit 加载基座模型
    bnb_4bit_quant_type="nf4",                  # 使用 NF4 数据类型
    bnb_4bit_compute_dtype=torch.bfloat16,      # 计算时的精度（反量化到 BF16 后计算）
    bnb_4bit_use_double_quant=True,             # 启用双重量化
)

# ---------- 加载基座模型（4-bit 量化）----------
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",                          # 自动分配到可用 GPU
    trust_remote_code=True,
)

# ---------- 为量化训练做准备 ----------
# 关键：将特定层转为可训练精度，并启用梯度检查点
model = prepare_model_for_kbit_training(model)

# ---------- 配置 LoRA ----------
lora_config = LoraConfig(
    r=16,                                       # QLoRA 通常用更大的秩（补偿量化误差）
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention 四个投影
        "gate_proj", "up_proj", "down_proj",      # FFN (SwiGLU)
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 输出: trainable params: ~42M || all params: ~6.7B (但以 4-bit 加载) || trainable%: ~0.6%
```

### 6.7 QLoRA vs LoRA 的本质区别

| 维度 | LoRA | QLoRA |
|------|------|-------|
| **基座模型精度** | FP16 / BF16 | NF4 (4-bit) |
| **LoRA 参数精度** | FP16 / BF16 | BF16 |
| **基座模型显存** | 2 bytes/参数 | ~0.5 bytes/参数（含量化常数）|
| **前向传播** | $h = W_0 x + \frac{\alpha}{r}BAx$ | $h = \text{dequant}(W_0^{NF4})x + \frac{\alpha}{r}(BA)^{BF16}x$ |
| **训练速度** | 快（无量化开销） | 慢 ~20%-30%（反量化开销） |
| **效果** | 理论上略好 | 与 LoRA 几无差距（原始论文结论） |

---

## 7. Alpaca：Instruction Tuning 的代表性案例

### 7.1 历史意义

Stanford Alpaca（2023年3月）是**首个以极低成本复现 GPT-3.5 级别指令跟随能力的开源模型**。虽然在今天已被 LLaMA-3、Qwen 等大幅超越，但其方法论对整个开源社区的影响是不可磨灭的。

### 7.2 Self-Instruct 数据生成管道

Alpaca 的核心创新在于用**模型生成训练数据**，解决了高质量指令数据的瓶颈问题：

```
阶段1：种子任务收集
    ↓  人工编写 175 条种子指令（覆盖写作、编程、问答、推理等）
    ↓
阶段2：指令扩展（Self-Instruct 流水线）
    ↓  将 175 条种子任务作为 few-shot 示例
    ↓  调用 text-davinci-003 生成新的指令（instruction）
    ↓  对每条新指令判断是否为分类任务 → 是则先生成类别 → 再生成具体 input
    ↓  调用 text-davinci-003 为每条指令生成 output
    ↓
阶段3：数据过滤与清洗
    ↓  去除与种子任务高度重复的指令
    ↓  去除包含不当内容的样本
    ↓  去除 output 过短或无意义的样本
    ↓  确保指令多样性（用 ROUGE-L 度量去重）
    ↓
阶段4：最终数据集
     ✓  52K 条高质量 instruction-input-output 三元组
```

**Self-Instruct 的数学直觉**：

可以将 Self-Instruct 看作一种数据增强（Data Augmentation），但它不是在特征空间做变换，而是在**语义空间**做扩展。教师模型 $M_T$（GPT-3.5）基于种子分布 $P_{seed}$ 生成扩展数据：

$$D_{synth} = \{(x, M_T(x)) | x \sim P_{expansion}(P_{seed})\}$$

学生模型 $M_S$（LLaMA 7B）在这个合成数据集上做监督微调，目标是逼近教师模型的行为函数。

### 7.3 训练细节

| 项目 | 详情 |
|------|------|
| 基座模型 | LLaMA 7B（Meta, 非开源协议） |
| 训练方式 | **全量微调**（所有 7B 参数更新） |
| 优化器 | AdamW |
| 学习率 | 2e-5 |
| Batch Size | 128（全局） |
| Epochs | 3 |
| 序列长度 | 512 tokens |
| 硬件 | 8×A100 80GB |
| 训练时长 | 约 3 小时 |
| 总成本 | ~$100（按 AWS 按需价格） |
| 损失函数 | 标准 Cross-Entropy（仅计算 output 部分） |

### 7.4 为什么 Alpaca 能成功？

1. **数据质量压倒数据数量**：52K 条经过仔细过滤的数据，效果远超更嘈杂的大数据集。这证明了数据策展（Data Curation）的重要性。
2. **教师模型的蒸馏**：GPT-3.5（text-davinci-003）经过 RLHF 对齐，其输出天然具有"有帮助的对话"风格，这种风格通过蒸馏传递给了 Alpaca。
3. **基座模型的迁移能力**：LLaMA 7B 的预训练已经赋予了强大的语言能力，只需少量的指令数据就能"唤醒"其指令遵循能力——这类似于能力涌现（Emergence）的另一种形式。
4. **成本极低，门槛极低**：$100 和 3 小时的数字极具冲击力，让整个社区意识到：指令微调不再是少数大公司的专利。

### 7.5 局限性（诚实的分析）

- **协议问题**：LLaMA 的非商用协议限制了 Alpaca 的实际应用（Stanford 后来撤下了模型权重）
- **蒸馏争议**：OpenAI 的服务条款禁止用其 API 输出训练竞争模型，Alpaca 处于法律灰色地带
- **幻觉存留**：未经 RLHF，Alpaca 的幻觉率显著高于 ChatGPT
- **安全未对齐**：没有偏好优化，容易被诱导生成有害内容
- **多样性有限**：175 条种子任务的覆盖面天然受限（偏重英语、编程、常识问答）

### 7.6 Alpaca 格式详解

```json
{
  "instruction": "将以下句子翻译成英文。",
  "input": "今天天气真好。",
  "output": "The weather is really nice today."
}
```

当 `input` 为空时（绝大多数情况）：

```json
{
  "instruction": "请解释什么是机器学习。",
  "input": "",
  "output": "机器学习是人工智能的一个分支，它使计算机能够从数据中学习和改进..."
}
```

**Prompt 模板**（训练时如何拼接）：

```
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```

当 `input` 为空时，`### Input:` 行被省略。这个模板格式后来被无数开源项目沿用和改进。

---

## 8. 常见数据格式

微调数据的格式因训练框架和目标而异，选择合适的数据格式直接影响模型的行为。

### 8.1 Alpaca 格式（单轮指令）

**结构**：

```json
{
  "instruction": "任务的描述",
  "input": "可选的上下文输入",
  "output": "期望的回复"
}
```

**适用场景**：单轮任务微调（翻译、摘要、分类等）  
**优点**：简单清晰，历史最悠久  
**缺点**：不支持多轮对话，不支持 system prompt

### 8.2 ShareGPT 格式（多轮对话）

**结构**：

```json
{
  "conversations": [
    {"from": "human", "value": "你好，请帮我写一个快速排序。"},
    {"from": "gpt", "value": "当然，以下是 Python 实现的快速排序：\n```python\ndef quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)\n```"},
    {"from": "human", "value": "能加上注释吗？"},
    {"from": "gpt", "value": "好的，这是带注释的版本：\n```python\ndef quicksort(arr):\n    # 递归终止条件：空数组或单元素数组已经有序\n    if len(arr) <= 1:\n        return arr\n    # 选择中间元素作为基准值\n    pivot = arr[len(arr) // 2]\n    # 分区：小于基准、等于基准、大于基准\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    # 递归排序左右子数组并拼接\n    return quicksort(left) + middle + quicksort(right)\n```"}
  ]
}
```

**适用场景**：通用对话模型（Chat Model）的训练  
**优点**：天然支持多轮对话，保留对话历史上下文  
**缺点**：格式略显冗余（`from` 字段设计不够标准化）

### 8.3 ChatML 格式（OpenAI / 业界标准）

**结构**：

```json
{
  "messages": [
    {"role": "system", "content": "你是一个有帮助的AI助手，回答应专业但通俗易懂。"},
    {"role": "user", "content": "什么是快速排序？"},
    {"role": "assistant", "content": "快速排序是一种分治排序算法..."},
    {"role": "user", "content": "它的时间复杂度是多少？"},
    {"role": "assistant", "content": "快速排序的平均时间复杂度为 O(n log n)..."}
  ]
}
```

**Role 类型**：

- `system`：系统级的指令，定义助手的行为和角色（如"你是一个法律顾问"）
- `user`：用户的输入
- `assistant`：模型的回复

**适用场景**：现代对话模型的标准格式，几乎所有主流模型（GPT-4、Claude、Qwen、DeepSeek）都支持  
**优点**：标准化程度高，支持 system prompt，支持多轮对话，生态支持最广  
**缺点**：对数据准备要求更严格（需确保多轮对话的一致性）

### 8.4 三种格式的本质区别

| 维度 | Alpaca | ShareGPT | ChatML |
|------|--------|----------|--------|
| **轮次** | 单轮 | 多轮 | 多轮 |
| **System Prompt** | ❌ 不支持 | ❌ 不支持 | ✅ 支持 |
| **数据复杂度** | 低 | 中 | 中-高 |
| **生态支持** | 旧项目多 | LLaMA-Factory, Vicuna | TRL, OpenAI, 主流新项目 |
| **训练目标** | 指令 → 回复的映射 | 多轮对话的上下文理解 | 角色扮演 + 多轮对话 |

> 💡 **选择建议**：2024 年后的新项目，**优先选择 ChatML 格式**。它不仅兼容性最好，而且支持 system prompt 的特性在构建生产级应用（如客服机器人、角色扮演助手）时至关重要。

---

## 9. Fine-tuning 的训练超参数：理论与实践

### 9.1 优化器选择

#### AdamW：事实标准

AdamW 是 Adam 的改进版本，将权重衰减（Weight Decay）从自适应学习率中解耦：

$$\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)$$

其中 $\lambda$ 为权重衰减系数。与原始 Adam 中的 L2 正则化不同，AdamW 的权重衰减直接作用于参数而非梯度，避免了对自适应学习率的干扰。

**为什么微调几乎总是用 AdamW？**

1. **自适应学习率**：不同参数需要不同的更新步长，Adam 的二阶矩估计天然处理了这个问题
2. **对超参数不敏感**：SGD 需要精细调节学习率和动量，而 AdamW 的默认参数 ($\beta_1=0.9,\beta_2=0.999,\epsilon=1e-8$) 在大多数情况下都能工作
3. **结合权重衰减**：AdamW 提供了有效的正则化，防止过拟合

### 9.2 学习率：最重要的超参数

学习率 $\eta$ 控制每次参数更新的步长，是微调中**最敏感的超参数**。

#### 按方法推荐的学习率范围

| 方法 | LR 下限 | LR 推荐 | LR 上限 | 原理 |
|------|---------|---------|---------|------|
| Full Fine-tuning | 1e-6 | 2e-5 ~ 5e-5 | 1e-4 | 需保守更新以保护预训练权重 |
| LoRA | 5e-5 | 1e-4 ~ 3e-4 | 5e-4 | 可训练参数少，可更大步长 |
| QLoRA | 1e-4 | 2e-4 ~ 5e-4 | 1e-3 | 量化引入噪声，大 LR 有一定正则化效果 |
| Prompt Tuning | 1e-3 | 5e-3 ~ 5e-2 | 1e-1 | 可训练参数极少（几千个），需激进学习率 |

#### 学习率调度策略

**Cosine Schedule** 是最主流的选择：

$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)$$

其中 $\eta_{max}$ 为峰值学习率（warmup 结束时），$\eta_{min}$ 通常设为 $\eta_{max}$ 的 1/10 或 0。

**其他调度策略**：

| 策略 | 公式 | 特点 |
|------|------|------|
| Cosine | $\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{t}{T}\pi))$ | 平滑衰减，最常用 |
| Linear | $\eta_t = \eta_{max} - \frac{t}{T}(\eta_{max} - \eta_{min})$ | 简单但后期可能过快衰减 |
| Constant | $\eta_t = \eta_{max}$ | 不衰减，适合极短训练 |
| Cosine with Restarts | 周期性重置 $\eta$ | 适合长训练，防止陷入局部最优 |

**Warmup** 的必要性：

$$\eta_t = \eta_{max} \cdot \frac{t}{t_{warmup}}, \quad t < t_{warmup}$$

训练初期，模型参数还在适应新数据分布，梯度的方差大、方向不稳定。Warmup 通过从小学习率逐步增大，避免了训练初期的不稳定更新。对于微调任务，warmup ratio 通常设为 0.03 ~ 0.1（即总步数的 3%~10%）。

### 9.3 Batch Size 与 Gradient Accumulation

**有效 Batch Size**：

$$\text{Effective Batch Size} = \text{Batch Size per GPU} \times \text{Num GPUs} \times \text{Gradient Accumulation Steps}$$

例如：每 GPU 4 个样本，1 张 GPU，累积 4 步 → 有效 Batch Size = 16。

**Gradient Accumulation** 的数学原理：

标准 SGD 的梯度更新：

$$\theta_{t+1} = \theta_t - \frac{\eta}{|B|}\sum_{i \in B} \nabla_\theta \mathcal{L}(x_i)$$

梯度累积将大 batch 拆分为 $k$ 个小 batch：

$$\theta_{t+1} = \theta_t - \frac{\eta}{|B|} \sum_{j=1}^{k} \sum_{i \in B_j} \nabla_\theta \mathcal{L}(x_i)$$

数学上等价于大 batch 训练，但显存需求降至 $\frac{1}{k}$。代价是训练时间增长（串行计算每个小 batch）。

> ⚡ **实用技巧**：如果显存允许 4 个样本，但想要有效 batch size = 16，设置 `gradient_accumulation_steps = 4`。训练速度和显存之间取得平衡。

### 9.4 Epochs 的选择

| 数据量 | 推荐 Epochs | 理由 |
|--------|------------|------|
| < 500 条 | 5 ~ 10 | 需要多次遍历才能充分学习 |
| 500 ~ 2K 条 | 3 ~ 5 | |
| 2K ~ 10K 条 | 2 ~ 3 | |
| 10K ~ 50K 条 | 1 ~ 2 | |
| > 50K 条 | 1（或 sub-epoch） | 数据量足够，多 epoch 易过拟合 |

> ⚠️ **过拟合指标**：如果训练 loss 持续下降但验证 loss 开始上升，应立即停止训练（early stopping）。

### 9.5 完整的训练超参数配置示例

```python
from transformers import TrainingArguments

# ---------- LoRA 训练参数 ----------
training_args = TrainingArguments(
    # ---- 核心参数 ----
    output_dir="./lora-finetuned-model",
    num_train_epochs=3,                      # 训练轮数
    per_device_train_batch_size=4,           # 每 GPU 的 batch size
    gradient_accumulation_steps=4,           # 梯度累积步数（有效 batch = 4 × 4 = 16）
    
    # ---- 优化器 ----
    optim="adamw_torch",                     # AdamW 优化器
    learning_rate=2e-4,                      # 学习率（LoRA 推荐 1e-4 ~ 3e-4）
    weight_decay=0.01,                       # 权重衰减（L2 正则化）
    max_grad_norm=1.0,                       # 梯度裁剪（防止梯度爆炸）
    
    # ---- 学习率调度 ----
    lr_scheduler_type="cosine",              # Cosine 衰减
    warmup_ratio=0.05,                       # 5% 的总步数用于 warmup
    
    # ---- 精度与效率 ----
    bf16=True,                               # 使用 BF16 混合精度（A100/H100）
    fp16=False,                              # 如果用 V100/A100 老卡则 fp16=True
    gradient_checkpointing=True,             # 梯度检查点（用计算换显存）
    
    # ---- 日志与保存 ----
    logging_steps=10,                        # 每 10 步记录一次 loss
    save_strategy="epoch",                   # 每个 epoch 保存一次检查点
    save_total_limit=2,                      # 最多保留 2 个检查点
    
    # ---- 评估 ----
    eval_strategy="steps",                   # 按步数评估
    eval_steps=200,                          # 每 200 步评估一次
    load_best_model_at_end=True,             # 训练结束后加载最佳模型
    metric_for_best_model="eval_loss",       # 以验证 loss 为最佳模型标准
    
    # ---- 其他 ----
    dataloader_num_workers=4,                # 数据加载的并行 worker 数
    seed=42,                                 # 随机种子（保证可复现）
    report_to="tensorboard",                 # 使用 TensorBoard 可视化
)
```

---

## 10. Fine-tuning 的资源要求

### 10.1 显存需求的精确估算公式

训练显存的精确估算：

$$M_{total} = \underbrace{P_{model}}_{\text{模型参数}} + \underbrace{P_{grad}}_{\text{梯度}} + \underbrace{P_{opt}}_{\text{优化器}} + \underbrace{P_{act}}_{\text{激活值}}$$

各分量展开：

| 分量 | Full FT (混合精度) | LoRA (FP16) | QLoRA (NF4) |
|------|-------------------|-------------|-------------|
| 模型参数 | $2P$ bytes (FP16) | $2P$ bytes (FP16) | $\approx 0.5P$ bytes (NF4) |
| 梯度 | $2P$ bytes (FP16) | $2P_{train}$ bytes | $2P_{train}$ bytes |
| 优化器 ($m_t$) | $4P$ bytes (FP32) | $4P_{train}$ bytes | $4P_{train}$ bytes |
| 优化器 ($v_t$) | $4P$ bytes (FP32) | $4P_{train}$ bytes | $4P_{train}$ bytes |
| 激活值 | $\mathcal{O}(B \cdot L \cdot d \cdot n_{layers})$ | 同左 | 同左 |
| **合计（不含激活）** | $12P$ bytes | $2P + 10P_{train}$ | $0.5P + 10P_{train}$ |

其中 $P$ 为总参数量，$P_{train}$ 为可训练参数量（LoRA 中 $P_{train} \approx 0.001P \sim 0.01P$）。

**代入 7B 模型验证**：

- Full FT：$12 \times 7\text{B} = 84\text{ GB}$（不含激活值，加上激活值约 112 GB）✅ 与经验公式吻合
- LoRA (r=8, $P_{train} \approx 0.001P$)：$2 \times 7\text{B} + 10 \times 7\text{M} \approx 14\text{ GB}$（加上激活值约 16 GB）✅
- QLoRA (NF4, $P_{train} \approx 0.001P$)：$0.5 \times 7\text{B} + 10 \times 7\text{M} \approx 3.5\text{ GB}$（加上激活值约 6 GB）✅

### 10.2 数据量建议（扩展版，含收敛行为分析）

微调所需的数据量并非绝对的——它与**模型规模、任务复杂度和期望效果**三者共同决定：

| 场景 | 收敛量 (极少) | 良好量 (推荐) | 饱和量 | 备注 |
|------|-------------|-------------|--------|------|
| 指令微调 | 1K | 5K ~ 52K | 100K+ | 多样性比数量重要——覆盖更多指令类型 > 增加同类指令的变体 |
| 领域知识注入 | 500 | 2K ~ 10K | 50K | 如果领域术语密度高，更需要质量（准确的术语使用）而非数量 |
| 风格/格式控制 | 200 | 1K ~ 5K | 10K | 本质上是一种"行为正则化"，少量高质量示例即可见效 |
| 单分类任务 | 100/类 | 500/类 | 5K/类 | 类间均衡 > 单类总量 |
| 多标签分类 | 200/类 | 1K/类 | 10K/类 | 标签共现的充分覆盖需要更多数据 |
| 数学推理 | 1K | 5K ~ 20K | 100K+ | 需要详细推理链（Chain-of-Thought），仅答案不够 |
| 代码生成 | 1K | 5K ~ 50K | 200K+ | 覆盖多种语言、多种算法模式 |
| 翻译（特定语言对） | 2K | 10K ~ 50K | 200K | 需覆盖多种领域和句式 |

> 🔬 **数据效率实验**：Zhou et al. (2023) 的 LIMA 实验表明，仅用 **1,000 条精心挑选的高质量样本**做 SFT，LLaMA 65B 就能展现出令人惊讶的指令遵循能力。这挑战了"越多越好"的传统观念，证明**数据质量的天花板远高于数据数量**。

> 🔑 **数据质量评估清单**：
> 1. 标注一致性：同一类型的指令，输出风格是否一致？
> 2. 事实正确性：输出中的事实性陈述是否准确？
> 3. 格式规范性：JSON/Markdown 等结构化输出是否有效？
> 4. 多样性：指令覆盖的类型是否足够广泛？
> 5. 去重率：训练集中有多少近乎重复的样本？

### 10.3 训练时间与成本估算

| 方法 | 硬件 | 模型 | 数据量 | 时间 | 成本 | 备注 |
|------|------|------|--------|------|------|------|
| Full FT | 8×A100 80GB | 7B | 52K | ~3h | ~$100 | Alpaca 的标准配置 |
| Full FT | 8×A100 80GB | 13B | 52K | ~6h | ~$200 | |
| Full FT | 32×A100 80GB | 70B | 52K | ~24h | ~$3,000+ | 需要大规模集群 |
| LoRA | 1×A100 40GB | 7B | 10K | ~2h | ~$20 | |
| LoRA | 1×A100 80GB | 13B | 10K | ~4h | ~$40 | |
| QLoRA | 1×RTX 4090 | 7B | 10K | ~3h | ~$0.50 (电费) | 本地消费级 |
| QLoRA | 1×A6000 48GB | 33B | 10K | ~8h | ~$20 | 单卡工作站 |
| QLoRA | 1×A100 80GB | 70B | 10K | ~8h | ~$40 | 云端 A100 |
| QLoRA | 2×RTX 4090 | 70B | 10K | ~12h | ~$1 (电费) | 双卡本地（需张量并行） |

### 10.4 微调后的模型压缩与部署

微调（尤其是全量微调）后，模型往往需要再次量化以降低推理成本，或者在边缘设备上部署。

#### 再次量化：GPTQ / AWQ / HQQ

微调后的模型可以进一步做 4-bit 或 8-bit 量化以加速推理：

| 方法 | 量化方式 | 是否需要校准数据 | 特点 |
|------|---------|----------------|------|
| **GPTQ** | 后训练逐层量化（Post-Training Quantization） | ✅ 需 ~128 条校准数据 | 基于 Hessian 矩阵的最优量化，精度损失小 |
| **AWQ** | 基于激活值的感知量化 | ✅ 需校准数据 | 保留 1% 的"重要"权重为 FP16，其余量化 |
| **HQQ** | 无需校准数据的快速量化 | ❌ 不需要 | 速度最快，精度略低于 GPTQ/AWQ |

**⚠️ 重要注意事项**：

1. **QLoRA 后再做 GPTQ 的风险**：对已经用 QLoRA（NF4 量化）训练过的模型再做 GPTQ，会导致**双重量化误差累积**。NF4 的量化误差 + GPTQ 的量化误差可能叠加，导致精度显著下降。

2. **推荐方案**：
   - 用 **FP16/BF16** 做 LoRA 微调
   - 导出合并后的 FP16 模型
   - 再用 GPTQ/AWQ 做一次性 4-bit 量化

   或直接使用 QLoRA 训练后的 NF4 模型进行推理（无需额外量化）。

#### 边缘端部署：ONNX / TensorRT / MLX

微调后的 LLM 需要在移动设备、笔记本或嵌入式设备上运行时，需要转换为专门的推理引擎格式：

| 工具 | 适用硬件 | 说明 |
|------|---------|------|
| **ONNX Runtime** | CPU / GPU（跨平台） | 微软的跨平台推理引擎，支持模型优化和量化 |
| **TensorRT-LLM** | NVIDIA GPU | 针对 NVIDIA GPU 的极致优化推理引擎 |
| **MLX** | Apple Silicon（M 系列芯片） | Apple 专为自家芯片优化的 ML 框架 |
| **llama.cpp** | CPU / 混合设备 | 基于 GGUF 格式的高效推理，支持量化和边缘部署 |
| **MNN / TNN** | 移动端（手机） | 阿里巴巴/腾讯的移动端推理引擎 |

**典型部署流水线**：

```
微调（QLoRA, FP16） → 合并 LoRA 权重 → FP16 模型
    ↓
GPTQ/AWQ 4-bit 量化 → 量化后模型
    ↓
ONNX / TensorRT / llama.cpp 格式转换
    ↓
部署到目标设备（服务器 / 笔记本 / 手机）
```

---

## 11. RLHF 与 DPO：从"会回答"到"说人话"

指令微调（SFT）让模型学会"回答"，但人类偏好不仅仅是正确——还涉及有用性、诚实性、无害性、语言风格等维度。RLHF 和 DPO 的目标就是对齐（Align）模型输出与人类偏好。

### 11.1 RLHF 的三阶段详解

#### 阶段一：Supervised Fine-Tuning (SFT)

与标准指令微调相同，用高质量人类标注的对话数据训练模型。这是整个流程的起点。

#### 阶段二：训练奖励模型 (Reward Model, RM)

**核心思想**：让人类标注者对不同回复进行偏好排序，训练一个"打分器"来模拟人类偏好。

**数据收集**：
对于同一个 prompt $x$，SFT 模型生成 $K$ 个不同的回复 $\{y_1, y_2, ..., y_K\}$，人类标注者对这些回复进行偏好排序，得到偏好对 $(y_w \succ y_l | x)$（$y_w$ 优于 $y_l$）。

**Bradley-Terry 偏好模型**：

人类偏好概率建模为：

$$P(y_w \succ y_l | x) = \frac{\exp(r_\phi(x, y_w))}{\exp(r_\phi(x, y_w)) + \exp(r_\phi(x, y_l))} = \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))$$

其中 $r_\phi(x, y)$ 为奖励模型对回复 $y$ 的标量评分，$\sigma$ 为 sigmoid 函数。

**奖励模型训练损失**：

$$\mathcal{L}_{RM}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l)) \right]$$

这是一个二分类交叉熵损失——让奖励模型学会判断哪个回复更好。

#### 阶段三：PPO 强化学习

目标：最大化奖励模型的评分，同时不要让模型偏离 SFT 模型太远（防止 reward hacking）。

$$\max_{\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)} \left[ r_\phi(x, y) - \beta \cdot \text{KL}(\pi_\theta(\cdot|x) \| \pi_{SFT}(\cdot|x)) \right]$$

- 第一项 $r_\phi(x, y)$：最大化奖励模型评分（让回复更符合人类偏好）
- 第二项 $\beta \cdot \text{KL}(\pi_\theta \| \pi_{SFT})$：KL 散度正则化，防止模型偏离太远而失去语言能力
- $\beta$：平衡系数，控制"讨好奖励模型"和"保持语言质量"之间的权衡

**PPO 的具体步骤**（简化）：

1. 从当前策略 $\pi_\theta$ 采样一批回复
2. 用奖励模型 $r_\phi$ 打分
3. 计算 advantage：$\hat{A}_t = r_t + \gamma r_{t+1} + ... - V(s_t)$
4. 用 PPO clipped objective 更新策略：

$$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left( \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} \hat{A}_t, \text{clip}\left(\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}, 1-\epsilon, 1+\epsilon\right) \hat{A}_t \right) \right]$$

#### RLHF 的主要挑战

| 挑战 | 说明 |
|------|------|
| **显存需求** | 需同时维护 SFT 模型、RM 模型、Actor 模型、Critic 模型——最多 4 个模型 |
| **Reward Hacking** | 模型学会"钻空子"拿高分而非真正变好（如生成啰嗦的回复来获得"完整性"高分） |
| **训练不稳定** | PPO 对超参数敏感，KL 系数 $\beta$ 的选择尤为关键 |
| **偏好数据成本** | 人类标注偏好对成本高昂（每条 $0.1 ~ $1） |

### 11.2 DPO：消去奖励模型的直接偏好优化

DPO（Rafailov et al., 2023）的核心创新在于**将 RLHF 的偏好优化问题重新参数化为一个可直接优化的分类损失**，不再需要显式训练一个奖励模型。

#### 数学推导（从 RLHF 到 DPO）

RLHF 的最优策略 $\pi^*$（在 KL 约束下）有解析形式：

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left(\frac{1}{\beta} r(x, y)\right)$$

其中 $Z(x) = \sum_y \pi_{ref}(y|x) \exp\left(\frac{1}{\beta} r(x, y)\right)$ 为配分函数。

反解出奖励函数：

$$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

将此代入 Bradley-Terry 偏好模型：

$$P(y_w \succ y_l | x) = \sigma\left( \beta \log \frac{\pi^*(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_{ref}(y_l|x)} \right)$$

注意 $\log Z(x)$ 抵消了！这意味着我们不需要计算这个困难的归一化常数。

#### DPO 损失函数

$$\mathcal{L}_{DPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]$$

**DPO 的直观理解**：

- $\log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}$：当前模型相对于参考模型对 chosen 回复的"偏好增量"
- $\log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}$：对 rejected 回复的"偏好增量"
- 差值越大，说明模型越喜欢 chosen 而非 rejected → loss 越低
- $\beta$：控制偏离 $\pi_{ref}$ 的惩罚强度（$\beta$ 越小，偏离惩罚越大）

| 维度 | RLHF (PPO) | DPO |
|------|-----------|-----|
| **奖励模型** | ✅ 需要独立训练 | ❌ 不需要 |
| **强化学习** | ✅ PPO | ❌ 纯监督学习 |
| **训练时模型数量** | 3-4 个 | 2 个（$\pi_\theta$ + $\pi_{ref}$，后者冻结） |
| **显存需求** | 极高 | 中等（约 LoRA 的 2 倍） |
| **训练稳定性** | 较低（RL 特有的不稳定性） | 较高（标准分类损失） |
| **实现复杂度** | 高（需 PPO 基础架构） | 低（~50 行代码） |
| **数学等价性** | — | 与 RLHF 在 Bradley-Terry 模型下有相同的最优解 |
| **偏好数据需求** | 同 | 同（chosen vs rejected 对） |
| **实际效果** | 经典方法，经大量验证 | 多数任务可比肩甚至超过 RLHF |

> 📊 **一个重要限制**：DPO 的数学等价性依赖于 Bradley-Terry 偏好模型的假设。如果人类偏好不满足这个模型（例如存在偏好的不可传递性），DPO 和 RLHF 的结果可能会分道扬镳。

---

## 12. Fine-tuning 的评估

### 12.1 评估维度

微调后模型的评估不仅仅是看一个数字，而是多维度、多层次的：

| 评估维度 | 评估内容 | 常用指标 | 工具/基准 |
|---------|---------|---------|---------|
| **指令遵循** | 是否能正确理解和执行指令 | 人工评估、GPT-4 评判 | AlpacaEval, MT-Bench |
| **事实准确性** | 输出的事实陈述是否正确 | 准确率、F1 | TruthfulQA, HaluEval |
| **安全性** | 是否拒绝有害请求 | 拒绝率、攻击成功率 | HarmBench, AdvBench |
| **通用能力** | 对原有能力的保持程度 | 各项 NLP benchmark 分数 | MMLU, HellaSwag, GSM8K |
| **生成质量** | 是否流畅、连贯、多样 | PPL, BLEU, ROUGE | 视任务而定 |
| **推理能力** | 多步推理是否正确 | 准确率 | GSM8K, MATH, BBH |

### 12.2 LLM-as-Judge：用 GPT-4 评估

由于开放式对话的质量难以用自动化指标衡量，业界广泛采用 **LLM-as-Judge** 方法：

```
Prompt:
Please act as an impartial judge and evaluate the quality of the response 
provided by an AI assistant to the user question displayed below. 
Your evaluation should consider factors such as the helpfulness, relevance, 
accuracy, depth, creativity, and level of detail of the response.

[Question]
{question}

[The Start of Assistant's Answer]
{answer}
[The End of Assistant's Answer]

Rate the response on a scale of 1 to 10.
```

- **优点**：与人类判断的相关性远高于 BLEU/ROUGE 等自动化指标
- **缺点**：成本较高（需调用商业 API），可能存在评判偏差（偏好某种风格的回复）

### 12.3 关键评估基准

| 基准 | 测量维度 | 数据量 | 说明 |
|------|---------|--------|------|
| **AlpacaEval 2.0** | 指令遵循 | 805 条 | GPT-4 Turbo vs 参考回复的对比胜率 |
| **MT-Bench** | 多轮对话能力 | 80 个多轮对话主题 | GPT-4 多维度评分 |
| **MMLU** | 知识广度 | 57 个学科，~14K 题 | 多项选择题 |
| **GSM8K** | 数学推理 | 8.5K 小学数学应用题 | 要求给出完整推理过程 |
| **HumanEval** | 代码生成 | 164 道编程题 | pass@k 度量 |
| **TruthfulQA** | 真实性 | 817 道题目 | 测量模型是否重复人类常见误解 |
| **IFEval** | 指令严格遵循 | 541 条 | 测量是否精确遵循格式约束 |

> 🔗 **与《Evaluation Harness 笔记》的关联**：各评估基准的详细定义、评估方式（Log-likelihood vs Generative）、统计显著性检验方法，以及 benchmark 污染检测等技术细节，请参见《Evaluation Harness 技术深度解析》第 3 节（Benchmark / Task / Metric 详解）和第 7 节（评估结果解读与统计检验）。

### 12.4 评估的陷阱与最佳实践

#### LLM-as-Judge 的系统性偏差

使用 LLM（如 GPT-4）作为评判者虽然方便，但存在多种已知偏差：

| 偏差类型 | 表现 | 缓解方法 |
|---------|------|---------|
| **位置偏差（Position Bias）** | 裁判模型倾向于给先看到的回复更高分 | 交换两个候选回复的位置，取平均分 |
| **长度偏差（Length Bias）** | 裁判模型偏爱更长的回复（即使内容冗余） | 在评分指令中明确注明"不考虑长度"；或控制回复长度在同一量级 |
| **自我增强偏差（Self-Enhancement Bias）** | GPT-4 给类似 GPT-4 风格的回复打高分 | 使用多个不同的裁判模型（如 Claude + GPT-4 + 人类） |
| **格式化偏差** | 回复格式（Markdown/列表/代码块）影响评分 | 统一回复格式后再提交给裁判 |

**最佳实践方案**：

```python
# 使用多裁判 + 位置交换来减轻偏差
judges = ["gpt-4", "claude-3", "gemini-pro"]
scores = []
for judge in judges:
    scores_original = judge_score(judge, question, answer_a, answer_b)
    scores_swapped = judge_score(judge, question, answer_b, answer_a)
    # 取两个位置的平均
    scores.append((scores_original + (1 - scores_swapped)) / 2)
final_score = np.mean(scores)  # 多裁判平均
```

#### 幻觉评估

微调后的模型可能产生看似合理但实际错误的内容，需要专门的评估工具：

| 工具 | 原理 | 适用场景 |
|------|------|---------|
| **HaluEval** | 构造包含幻觉的 QA 对，评估模型能否识别 | 通用幻觉检测 |
| **SelfCheckGPT** | 对同一 prompt 多次采样，检查回答间的一致性 | 无需参考答案的幻觉检测 |
| **FACET** | 对模型生成中每个事实性声明独立验证 | 细粒度事实性评估 |
| **RAGAS** | 结合检索评估生成内容的忠实性（Faithfulness） | RAG 系统的幻觉评估 |

#### 多语言评估

如果微调目标是非英语语言（如中文、日语、阿拉伯语），必须使用针对性的评估基准：

| 语言 | 推荐基准 | 说明 |
|------|---------|------|
| **中文** | C-Eval, CMMLU, M3KE, MMCU | 涵盖学科知识、常识推理、医学等领域 |
| **日语** | JMMLU, JCommonSenseQA | 日语版本的多任务语言理解 |
| **阿拉伯语** | Arabic MMLU, ARBERT | 针对阿拉伯语的基准 |
| **多语言** | Flores-200（翻译）, XCOPA（推理）, XNLI（自然语言推理） | 跨语言评估 |

> ⚠️ **常见陷阱**：在多语言微调中，仅用 MMLU（英文）评估可能会高估模型的实际能力。建议始终在目标语言对应的基准上单独评估。

#### 评估的完整性检查清单

- [ ] 是否在多个基准上评估（不只是 1 个）？
- [ ] 是否使用了**与训练数据不重叠**的评估数据？（防止数据泄露）
- [ ] 是否评估了通用能力的保持程度（不仅仅是目标任务的提升）？
- [ ] 对于开放式生成，是否使用了至少 2 个裁判模型？
- [ ] 是否排除了位置偏差（交换 A/B 位置再测）？
- [ ] 是否记录了评估时的超参数（temperature, top_p 等）？
- [ ] 如果微调语言 ≠ 英语，是否在目标语言的基准上评估了？

---

## 13. 常见问题与踩坑指南

### 13.1 灾难性遗忘 (Catastrophic Forgetting)

**形式化定义**：设 $\mathcal{T}_{pre}$ 为预训练任务的分布，$\mathcal{T}_{ft}$ 为微调任务的分布。灾难性遗忘指微调后的模型在 $\mathcal{T}_{pre}$ 上的性能显著下降：

$$\mathbb{E}_{x \sim \mathcal{T}_{pre}}[\text{Perf}(f_{\theta_{FT}}, x)] \ll \mathbb{E}_{x \sim \mathcal{T}_{pre}}[\text{Perf}(f_{\theta_0}, x)]$$

**根本原因**：梯度更新 $\theta_{FT} = \theta_0 - \eta \sum_{t} \nabla_\theta \mathcal{L}_{ft}$ 的方向与预训练的最优点正交（或接近正交），导致模型在预训练分布上"漂移"。

**缓解方案（按优先级排序）**：

1. **使用 PEFT**（LoRA/QLoRA）——冻结 $\theta_0$，遗忘程度最低
2. **数据回放（Data Replay）**——在微调数据中混入 5%-10% 的通用数据
3. **弹性权重巩固（EWC, Elastic Weight Consolidation）**——在损失函数中加入对重要参数的惩罚项：

   $$\mathcal{L}_{EWC} = \mathcal{L}_{ft} + \frac{\lambda}{2} \sum_i F_i (\theta_i - \theta_{0,i})^2$$

   其中 $F_i$ 为 Fisher 信息矩阵的对角元（衡量参数 $i$ 对预训练任务的重要性）

4. **适当降低学习率**——减少每次更新的破坏力
5. **早停（Early Stopping）**——在验证集 perplexity 开始上升时停止

### 13.2 过拟合

**诊断信号**：

$$\text{Train Loss} \downarrow \downarrow, \quad \text{Val Loss} \uparrow \quad \Longrightarrow \quad \text{过拟合}$$

**量化指标**：微调后模型的训练 loss 和验证 loss 的差距扩大。

**预防策略**：

| 策略 | 效果 | 代价 |
|------|------|------|
| 减少 epochs | ⭐⭐⭐ | 无 |
| 增加 LoRA dropout (0.05 → 0.1) | ⭐⭐ | 轻微降低训练速度 |
| 数据增强（同义改写、反向翻译） | ⭐⭐⭐ | 需额外处理 |
| 更小的 LoRA rank ($r=16 \to r=4$) | ⭐⭐ | 可能降低效果上限 |
| 增加 weight decay | ⭐ | 影响收敛速度 |
| Mixout（以概率 $p$ 将参数重置为预训练值） | ⭐⭐ | 实现较复杂 |

### 13.3 数据质量陷阱

**标注噪声的量化影响**：

研究表明，标注数据中即使只有 10% 的错误标签，也会导致模型效果下降 5-15%（取决于任务难度）。对于少样本场景（< 500 条），这个影响尤为严重。

**数据质量检查清单**：

- [ ] 指令和回复的语言是否一致（中英混合？）
- [ ] 回复是否真正回答了指令问的问题？
- [ ] 格式是否一致（不要混用 JSON 和纯文本）？
- [ ] 是否有明显的错误信息（日期、人名、事实）？
- [ ] 数据集中是否有大量近似重复的样本？
- [ ] 多轮对话中，上下文是否连贯？
- [ ] 是否存在偏见内容（性别、种族、地域等）？

**数据泄露的危害**：

如果训练集中混入了评估集（如无意中包含了 MMLU 的部分题目），会导致评估分数虚高但实际能力并未提升——这是研究中的严重错误，会导致论文被撤稿。

### 13.4 LoRA 调参深度指南

**$r$（秩）的选择**：

$$\text{参数量}(r) \propto r, \quad \Delta\text{效果}(r_1 \to r_2) \text{ 通常随 } r \text{ 增长而递减}$$

| 数据量 | 推荐 $r$ | 原理 |
|--------|---------|------|
| < 500 条 | 4 | 防止过拟合 |
| 500 ~ 2K | 8 | 性价比最优 |
| 2K ~ 10K | 8 ~ 16 | |
| 10K ~ 50K | 16 ~ 32 | 数据丰富，可以学习更复杂的模式 |
| > 50K | 32 ~ 64 | 大数据集可以支撑更高的秩 |

**$r=4$ 和 $r=64$ 的效果真的差很多吗？**

在大多数指令微调场景中，$r=8$ 和 $r=64$ 的最终效果差距在 **1% 以内**。这就是 LoRA 论文的核心发现——任务相关的权重更新确实集中在非常低维的子空间中。盲目增大 $r$ 是浪费计算资源。

### 13.5 其他工程注意事项

- **Chat Template 是隐形的杀手**：每个基座模型（LLaMA-2, LLaMA-3, Qwen, ChatGLM）使用不同的特殊 token。用错 template 会导致模型无法正确区分 user/assistant 的边界，输出混乱。务必查文档确认。
- **EOS Token 不匹配**：如果训练时 EOS token 设置错误，模型不知道何时停止生成，会反复循环或产生无意义内容。
- **训练/推理时的 attention mask 配置**：SFT 时通常只对 assistant 的回复计算 loss（mask 掉 user 输入部分）。如果 mask 配置错误，模型会模仿 user 的提问方式而非学习回答。
- **去重的重要性**：训练集中有大量重复数据会导致模型输出重复（Repetition Problem）。一项研究发现，仅 5% 的数据重复率就能导致模型在 10% 的生成中出现重复循环。

### 13.6 法律与安全陷阱

#### 数据版权与合规

微调过程中使用的训练数据可能涉及版权和法律风险：

1. **使用商业模型输出蒸馏的风险**：如果你用 GPT-4 / Claude 的 API 生成训练数据来微调开源模型，可能违反服务条款。OpenAI 和 Anthropic 的 ToS 通常禁止用 API 输出来训练与自身竞争的模型。法律上处于灰色地带（当前缺乏明确判例），但实践中建议：
   - 仔细阅读所用 API 的 ToS
   - 优先使用开源模型（如 LLaMA-3、Qwen、DeepSeek）生成合成数据
   - 使用人工标注数据（最安全，但成本最高）

2. **数据中不要包含个人身份信息（PII）**：在微调数据中使用真实用户的隐私信息（姓名、电话、邮箱、地址等）可能违反 GDPR、CCPA、中国《个人信息保护法》等法律法规。处理建议：
   - 对数据集进行 PII 检测和脱敏
   - 使用工具（如 Microsoft Presidio、Faker）生成合成 PII 替代真实信息
   - 记录数据来源，确保获得授权使用

#### 安全对齐遗忘（Safety Alignment Forgery）

**现象**：模型经过微调后，原有的安全对齐（如拒绝生成有害内容的行为）会显著退化。

**原因**：微调数据的分布通常不包含拒绝类样本，模型在 SFT 过程中逐渐"遗忘"了拒绝有害请求的能力：

$$P_{FT}(\text{refuse} | \text{harmful request}) \ll P_{base}(\text{refuse} | \text{harmful request})$$

**缓解方法**：

| 方法 | 说明 | 效果 |
|------|------|------|
| **安全数据混合** | 在 SFT 数据中混入 5-10% 的安全拒绝样本 | ⭐⭐⭐ 最有效 |
| **DPO 安全对齐** | 先用 SFT 微调领域知识，再用 DPO 恢复安全对齐 | ⭐⭐⭐ |
| **安全滤波器** | 在模型前部署一个独立的安全检测分类器 | ⭐⭐（外挂方案） |
| **LoRA 模块分离** | 领域知识 LoRA 和安全 LoRA 分开训练，推理时同时加载 | ⭐⭐（灵活性好） |

**经验法则**：**任何微调都应对齐遗忘做基线测试**——在微调前后分别对同一组有害 prompt 测试，确保拒绝率没有显著下降。

#### 输出内容合规

微调后的模型如果部署在面向用户的场景中（客服、教育、医疗等），输出内容可能需要满足特定的合规要求：

- **行业合规**：医疗领域的 HIPAA、金融领域的 SOX / PCI DSS 等对模型输出有特殊要求
- **内容安全**：中国《生成式人工智能服务管理暂行办法》要求模型不生成违法内容
- **偏见与公平性**：微调数据中的偏见可能被放大，导致模型输出具有歧视性

> 💡 **实践建议**：生产环境中，**永远不要在用户和微调后的模型之间不加防护**。始终在模型前部署内容安全过滤器（如 NeMo Guardrails、Guardian AI、Llama Guard），并设置人工抽检流程。

---

## 14. 实用工具与框架

| 工具/框架 | 核心功能 | 最适合的场景 | 备注 |
|----------|---------|-------------|------|
| **Hugging Face Transformers** | 模型加载、训练、推理 | 所有场景的基础 | 生态核心，必学 |
| **Hugging Face PEFT** | LoRA/QLoRA/Adapter/Prompt Tuning | PEFT 的标准实现 | 与 Transformers 无缝集成 |
| **Hugging Face TRL** | SFT、DPO、RLHF、PPO | 对齐训练（DPO/RLHF） | 封装了 SFTTrainer, DPOTrainer |
| **bitsandbytes** | 4-bit/8-bit 量化 | QLoRA 的量化后端 | QLoRA 依赖此库 |
| **Axolotl** | 一站式微调框架 | 新手或快速实验 | YAML 配置，无需写代码 |
| **LLaMA-Factory** | 国产微调框架，Web UI | 中文用户、可视化操作 | 支持几乎所有主流模型 |
| **Unsloth** | 加速 LoRA/QLoRA 2-5 倍 | 追求训练速度 | 手写 CUDA kernel，真·加速 |
| **Flash Attention 2** | 加速 Attention，降低显存 | 所有长序列训练 | 将显存从 $O(n^2)$ 降至 $O(n)$ |
| **DeepSpeed ZeRO** | 分布式训练优化 | 多卡训练 | ZeRO-1/2/3 逐级降低显存 |
| **vLLM** | 高性能推理引擎 | 微调后的模型部署 | PagedAttention，吞吐量极高 |
| **mergekit** | 模型合并框架 | 合并多个微调后的权重/LoRA | 支持 TIES-Merging、DARE、Linear Merge |
| **Wandb / TensorBoard** | 训练可视化和监控 | 所有训练任务 | 监控 loss 曲线、梯度分布等 |
| **OpenRLHF / veRL** | RLHF 分布式框架 | 大规模 RLHF 训练 | Ray 后端，支持千卡集群 |

---

## 15. 总结与速查

### 15.1 按场景选方法

| 你想做什么？ | 推荐方法 | 核心配置 | 理由 |
|-------------|---------|---------|------|
| 消费级显卡微调 7B | **QLoRA** | r=8~16, NF4 | 6GB 显存即可，单卡 RTX 3060 都能跑 |
| A100 单卡微调 7B | **LoRA** | r=8~16, BF16 | 无量化开销，比 QLoRA 快 ~20% |
| 消费级微调 70B | **QLoRA** | r=16~64, NF4+Double Quant | A6000/4090 48GB 可行 |
| 多卡集群追求极致 | **Full Fine-tuning + DeepSpeed ZeRO-3** | 混合精度, 大 batch | 显存/算力成本极高，一般不建议 |
| 定制输出格式/风格 | **QLoRA** | r=4~8, 500-2K 高质量样本 | 数据需求小，重点在数据质量 |
| 安全对齐/降低幻觉 | **SFT → DPO** | SFT(QLoRA) + DPO(QLoRA) | 两步走，先学会回答再学会说人话 |
| 快速实验/原型验证 | **LLaMA-Factory + QLoRA** | Web UI 点选 | 5 分钟上手，不用写代码 |
| 追求训练速度 | **Unsloth + LoRA** | 手写 CUDA kernel | 比标准 LoRA 快 2-5 倍 |

### 15.2 核心概念关系图

```
                        Pre-training (自监督)
                        ／                    ＼
                       ／                      ＼
                Fine-tuning                Instruction Tuning
               （任务特定微调）               （指令微调 / SFT）
               ／         ＼                      │
              ／           ＼               ┌─────┴──────┐
       Full FT          PEFT               │             │
     （全量微调）    （参数高效微调）       RLHF          DPO
                         │              （PPO+RM）   （直接偏好）
                    ┌────┼────┐
                    │    │    │
                 LoRA  QLoRA Adapter/Prefix/Prompt
```

### 15.3 一句话精华

1. **迁移学习是灵魂**：预训练模型提供了通用智能的基石，微调将其转化为专用能力
2. **PEFT 是实践首选**：除非你有充分的理由（和大批 GPU），否则用 LoRA/QLoRA
3. **数据质量压倒数量**：5K 精心策展的数据 > 50K 嘈杂数据（LIMA 实验证明了这一点）
4. **LoRA 的 $r=8$ 几乎总是够用**：权重更新确实集中在非常低维的子空间
5. **QLoRA 让大模型微调民主化**：一张 4090 就能微调 70B 模型
6. **SFT 教会"回答"，DPO/RLHF 教会"说人话"**：两者缺一不可
7. **Chat Template 是隐藏的雷**：用错 template 会毁掉所有努力
8. **评估要全面**：不要只看一个指标——指令遵循、事实性、安全性、通用能力都要看

---

## 16. 多模态模型的微调要点

随着 LLaVA、Qwen-VL、Fuyu、InternVL 等视觉语言模型（VLM）的流行，微调早已不仅限于纯文本模型。本节介绍多模态微调的核心概念。

### 16.1 视觉语言模型的典型架构

大多数现代 VLM 采用 **投影器桥接（Projector Bridge）** 架构：

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│  视觉编码器   │───→│   投影器     │───→│   LLM 基座模型   │
│  (冻结/解冻)  │    │  (可训练)    │    │  (LoRA 微调)    │
└─────────────┘    └──────────────┘    └─────────────────┘
      ↑                     ↑                    ↑
   图像输入          视觉→文本映射           文本指令+图像特征
```

| 组件 | 典型实现 | 是否可训练 | 说明 |
|------|---------|-----------|------|
| **视觉编码器** | CLIP ViT-L/14, SigLIP, InternViT | ❌ **通常冻结** | 已学会通用视觉特征，微调成本高 |
| **投影器** | MLP (2-3层), Q-Former (BLIP-2) | ✅ 可训练 | 将视觉特征映射到 LLM 的 embedding 空间 |
| **LLM 基座模型** | LLaMA-3, Qwen-2, DeepSeek | ✅ LoRA 微调 | 负责理解指令并生成文本回复 |

### 16.2 微调策略

#### 标准方案：投影器 + LoRA on LLM

这是目前最主流的 VLM 微调方式：

1. **冻结视觉编码器**（ViT）：视觉编码器参数量大（~300M-1B），训练成本高且易过拟合
2. **训练投影器**：MLP 参数极少（~10M-100M），作为视觉和文本的桥接
3. **LoRA 微调 LLM**：让 LLM 学会理解图像特征并生成相关回复

> 📊 **经验结论**：绝大多数 VLM 微调项目中，冻结视觉塔、仅训练投影器和 LLM 的 LoRA，就能达到与全量微调 95%+ 的效果。除非目标领域与预训练领域差异极大（如医疗影像、卫星图像），否则不需要解冻视觉编码器。

#### 特殊情况：何时需要解冻视觉编码器？

- **领域转移极大**（如自然图像 → 医疗 CT / 卫星遥感 / 显微图像）
- **输入格式变化**（如从矩形图像到全景图、多视角图）
- **视觉任务复杂**（如细粒度物体检测、文档布局分析）

此时可考虑：
1. **先冻结训练投影器 + LLM LoRA**（基础对齐）
2. **再用极小学习率（~1e-6）解冻视觉编码器做精细调整**

### 16.3 多模态微调的数据格式

多模态对话数据通常扩展了 ChatML 格式，支持多种类型的内容：

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "url": "https://example.com/photo.jpg"},
        {"type": "text", "text": "请描述这张图片中的场景。"}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "这张照片拍摄于一个阳光明媚的公园，画面中央有一棵盛开的樱花树，树下有几个人坐在长椅上休息。背景中可以看到远处的城市天际线。"}
      ]
    },
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "这张图片是什么季节拍摄的？"}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "从樱花盛开来判断，应该是春季拍摄的。"}
      ]
    }
  ]
}
```

**多轮图文交错对话的 key 要点**：

- 每条消息的 `content` 为数组，可包含多个 `type` 的混合
- 图片通常以 URL、base64、或本地路径引用
- 不同框架的字段名有差异：
  | 框架 | 图片字段 |
  |------|---------|
  | LLaVA | `"image": "image_path"`（同层） |
  | Qwen-VL | `{"type": "image", "image": "url"}` |
  | OpenAI Vision | `{"type": "image_url", "image_url": {"url": "..."}}` |
  | InternVL | `{"role": "user", "content": "<image>\\n文本描述"}`（特殊 token） |

### 16.4 多模态微调的工程注意事项

1. **图像分辨率与分块**：高分辨率图像需分块（如 LLaVA-NeXT 将图像分为 grid），训练时需注意内存管理
2. **位置编码**：如果图像分辨率与预训练时不同，需调整位置编码（插值或动态缩放）
3. **多图像支持**：支持批量输入的 VLM 需处理图像与文本的对齐（文本中对应 `<image>` token 的位置）
4. **数据加载**：图像解码是 I/O 瓶颈，建议多进程预加载和缓存

### 16.5 推荐工具与框架

| 框架 | 支持模型 | 特点 |
|------|---------|------|
| **LLaVA** | LLaVA-1.5/1.6/NeXT | 最经典的 VLM 微调框架 |
| **LLaMA-Factory** | LLaVA, Qwen-VL, InternVL | 统一的多模态微调界面 |
| **SWIFT** | 50+ VLM 模型 | 阿里开源，支持多模态微调 |
| **Axolotl** | LLaVA, Qwen-VL | 支持多模态的 YAML 配置 |

---

## 17. 三份笔记的串联：端到端管线

### 17.1 整体流程

本系列三份笔记（Fine-tuning、Alignment、Evaluation）覆盖了从基座模型到可部署模型的完整流程：

```
原始语料（预训练数据）→ 预训练（不在本系列覆盖范围）
    │
    ▼
基座模型（Base Model）
    │
    ├──→ 【Finetuning 笔记】
    │     │  SFT（指令微调） / LoRA / QLoRA
    │     │  数据格式：Alpaca / ShareGPT / ChatML
    │     │  目标：教会模型"回答问题"
    │     │  产出：微调后的模型（Instruct Model / Chat Model）
    │     ▼
    │
    ├──→ 【Alignment 笔记】
    │     │  DPO / RLHF / GRPO / RLAIF
    │     │  数据格式：偏好对（chosen / rejected）
    │     │  目标：教会模型"说人话"（有帮助、诚实、无害）
    │     │  产出：对齐后的模型（Aligned Model）
    │     ▼
    │
    └──→ 【Evaluation 笔记】
          │  lm-evaluation-harness / 人工评估
          │  基准：MMLU, GSM8K, HumanEval, TruthfulQA 等
          │  目标：衡量能力边界，指导下一轮迭代
          │  产出：评估报告 → 返回 Finetuning/Alignment 继续优化
    
最终产出：可用于部署的高质量对齐模型
```

**三份笔记的依赖关系**：

- 阅读顺序建议：**Finetuning → Alignment → Evaluation**
- Finetuning 是基础——教会模型完成任务
- Alignment 是进阶——优化模型的行为和价值观
- Evaluation 是所有环节的"考试"——没有评估就无法判断微调和对齐的效果

### 17.2 常见失败模式诊断清单

以下清单总结了微调 + 对齐 + 评估全流程中常见的异常模式，帮助你快速定位问题：

| 观察到的现象 | 可能原因 | 涉及笔记 | 建议排查方向 |
|------------|---------|---------|-------------|
| 微调后 MMLU 下降 1-2%，但对话质量提升 | **正常对齐税** | Finetuning §8.3 + Alignment §8.3 | 这是预期行为，无需过度干预。如果下降超过 3%，需检查数据质量 |
| 微调后 GSM8K 大幅下降（>10%），但 MMLU 不变 | **数学推理能力遗忘** | Finetuning §13.1 | 检查微调数据中是否缺少数学/推理样本；考虑混入 5-10% 的推理数据做回放 |
| DPO 后拒绝率过高（正常问题也被拒） | **β 太大** 或 **偏好数据中 reject 样本过多** | Alignment §4.3 | 降低 β（如 0.1→0.05）；检查偏好数据中 rejected 回复是否过于苛刻 |
| DPO 后仍然拒绝有害请求不足 | **安全对齐遗忘** | Alignment §6.6 + Finetuning §13.6 | 在 SFT 数据中混入 5-10% 的安全拒绝样本；或单独用安全数据做 DPO |
| TruthfulQA 分数低 | **幻觉严重** | Alignment §8.4 + Eval §7.4 | 检查模型校准度（ECE）；加入诚实性偏好数据重新 DPO |
| 同一模型两次评估结果差异大 | **可复现性不足** | Eval §8.7 | 检查 seed 是否固定、GPU 精度设置是否一致、lm-eval 版本是否相同 |
| 某个 benchmark 分数异常高 | **数据污染** | Eval §8.1 | 做 n-gram 污染检测；换用 LiveBench 等动态基准验证 |
| 两个模型分数接近但结论反转 | **统计不显著** | Eval §7.1 | 做 McNemar 检验或 Bootstrap 置信区间，确认差异是否显著 |
| LLM-as-Judge 结果与人工评估不一致 | **裁判模型偏差** | Finetuning §12.4 + Alignment §8.4 | 使用多裁判（Claude + GPT-4 + 人类）；交换 A/B 位置取平均 |
| 微调后生成重复内容 | **训练数据去重不足** | Finetuning §13.5 | 检查训练数据中是否有大量近似重复样本 |

### 17.3 统一实验记录模板

在做微调 → 对齐 → 评估的全流程实验时，建议使用以下模板统一记录实验信息，确保可复现性：

```markdown
## 实验记录

### 基础信息
| 项目 | 值 |
|------|-----|
| 实验编号 | |
| 实验目的 | |
| 日期 | |

### 模型信息
| 项目 | 值 |
|------|-----|
| 基座模型 | |
| 微调方法 | Full FT / LoRA / QLoRA |
| 对齐方法 | SFT / DPO / RLHF / GRPO |
| 模型精度 | FP16 / BF16 / NF4 |

### 数据信息
| 项目 | 值 |
|------|-----|
| 微调数据集 | 名称、规模、格式 |
| 偏好数据集 | 名称、规模、来源 |
| 评估数据集 | MMLU / GSM8K / HumanEval / ... |

### 训练超参数
| 项目 | 值 |
|------|-----|
| 学习率 | |
| Batch Size | |
| LoRA rank / α | |
| DPO β | |
| 训练 Epochs | |
| 随机种子 | |

### 硬件与软件环境
| 项目 | 值 |
|------|-----|
| GPU 型号与数量 | |
| PyTorch 版本 | |
| Transformers 版本 | |
| TRL / PEFT 版本 | |
| lm-eval 版本 | |
| CUDA 版本 | |

### 评估结果
| Task | Metric | Score | 与基线差距 | 备注 |
|------|--------|-------|-----------|------|
| MMLU | acc | | | |
| GSM8K | exact_match | | | |
| HumanEval | pass@1 | | | |
| TruthfulQA | acc | | | |
| MT-Bench | GPT-4 score | | | |

### 备注 / 观察到的异常
```

---

## 参考资源

### 核心论文

- 📄 **LoRA**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al., 2021
- 📄 **QLoRA**: [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Dettmers et al., 2023
- 📄 **DPO**: [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) — Rafailov et al., 2023
- 📄 **InstructGPT**: [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — Ouyang et al., 2022
- 📄 **Self-Instruct**: [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560) — Wang et al., 2022
- 📄 **LIMA**: [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) — Zhou et al., 2023
- 📄 **Intrinsic Dimensionality**: [Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning](https://arxiv.org/abs/2012.13255) — Aghajanyan et al., 2020
- 📄 **Adapter**: [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751) — Houlsby et al., 2019
- 📄 **Prefix Tuning**: [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190) — Li & Liang, 2021
- 📄 **Prompt Tuning**: [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691) — Lester et al., 2021

### 工具与项目

- 🦙 **Alpaca**: [Stanford Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html)
- 🛠️ **PEFT**: [Hugging Face PEFT](https://huggingface.co/docs/peft)
- 🏋️ **TRL**: [Hugging Face TRL](https://huggingface.co/docs/trl)
- 🏭 **LLaMA-Factory**: [GitHub - hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- ⚡ **Unsloth**: [GitHub - unslothai/unsloth](https://github.com/unslothai/unsloth)
- 📊 **Axolotl**: [GitHub - OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)
- 🔢 **bitsandbytes**: [GitHub - bitsandbytes-foundation/bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)

---

## 附录 A：从零开始微调一个中文法律问答模型（QLoRA + ChatML）

本附录提供一个端到端的最小示例，从数据准备到训练到推理部署，帮助你建立完整的微调工作流认知。

### A.1 数据准备（10 条示例）

创建 `legal_data.jsonl`，每行一个 ChatML 格式的样本：

```jsonl
{"messages": [{"role": "user", "content": "什么是劳动合同的试用期？"}, {"role": "assistant", "content": "劳动合同的试用期是用人单位和劳动者在劳动合同中约定的相互考察期。根据《劳动合同法》第十九条，试用期最长不得超过6个月。同一用人单位与同一劳动者只能约定一次试用期。试用期包含在劳动合同期限内。"}]}
{"messages": [{"role": "user", "content": "租房合同到期后不续租需要提前通知吗？"}, {"role": "assistant", "content": "根据《民法典》第七百三十四条，租赁期限届满，承租人继续使用租赁物，出租人没有提出异议的，原租赁合同继续有效，但是租赁期限为不定期。如果您不打算续租，建议按合同约定的提前通知期（通常为15-30天）书面告知出租人。若无约定，也建议提前合理时间通知，以便出租人寻找下一位租客。"}]}
{"messages": [{"role": "user", "content": "交通事故中如何划分责任？"}, {"role": "assistant", "content": "交通事故的责任划分主要依据《道路交通安全法》第七十三条。公安机关交通管理部门根据当事人的行为对发生交通事故所起的作用以及过错的严重程度，确定当事人的责任。主要分为：全部责任、主要责任、同等责任、次要责任和无责任五种。具体划分需要考虑：违反交通规则的程度、是否存在故意行为、因果关系等因素。"}]}
{"messages": [{"role": "user", "content": "网购商品出现质量问题，消费者有哪些权利？"}, {"role": "assistant", "content": "根据《消费者权益保护法》第二十四条，经营者提供的商品或者服务不符合质量要求的，消费者可以依照国家规定、当事人约定退货，或者要求经营者履行更换、修理等义务。网购商品适用七日无理由退货（《消费者权益保护法》第二十五条）。同时根据《电子商务法》，电商平台在特定情况下需承担连带责任。"}]}
{"messages": [{"role": "user", "content": "什么是知识产权？包括哪些类型？"}, {"role": "assistant", "content": "知识产权是指人们对其智力劳动成果所享有的专有权利。主要包括：(1)著作权（版权），保护文学、艺术和科学作品；(2)专利权，保护发明、实用新型和外观设计；(3)商标权，保护商品和服务的标识；(4)商业秘密，保护不为公众所知悉的技术和经营信息。我国的知识产权法律体系由《著作权法》《专利法》《商标法》《反不正当竞争法》等组成。"}]}
{"messages": [{"role": "user", "content": "公司可以随意调整员工的工资吗？"}, {"role": "assistant", "content": "根据《劳动合同法》第三十五条，用人单位与劳动者协商一致，可以变更劳动合同约定的内容，包括工资标准。因此，公司不能单方面随意调整员工工资。调薪需要双方协商一致并签订书面变更协议。如果公司未经协商擅自降薪，劳动者可以向劳动监察部门投诉或申请劳动仲裁。"}]}
{"messages": [{"role": "user", "content": "什么是遗嘱？有哪几种形式？"}, {"role": "assistant", "content": "遗嘱是自然人生前依法处分其个人财产，并于其死亡时发生效力的法律行为。根据《民法典》继承编，遗嘱的形式包括：(1)自书遗嘱，由遗嘱人亲笔书写并签名注明年月日；(2)代书遗嘱，应当有两个以上见证人在场见证；(3)打印遗嘱；(4)录音录像遗嘱；(5)口头遗嘱，仅在危急情况下使用；(6)公证遗嘱。不同形式的遗嘱对见证人、签名、日期等有不同要求。"}]}
{"messages": [{"role": "user", "content": "借条和欠条有什么区别？"}, {"role": "assistant", "content": "借条和欠条的主要区别在于：(1)证明的法律关系不同：借条证明借贷关系，欠条证明欠款关系（可能源于买卖、劳务等多种原因）；(2)诉讼时效不同：有还款期限的借条，诉讼时效为还款期限届满后3年；欠条的诉讼时效为出具欠条之日起3年。无还款期限的借条，诉讼时效最长可达20年（从出借之日起算）。建议在民事经济往来中明确使用借条并约定还款日期。"}]}
{"messages": [{"role": "user", "content": "什么是正当防卫？"}, {"role": "assistant", "content": "根据《刑法》第二十条，正当防卫是指为了使国家、公共利益、本人或者他人的人身、财产和其他权利免受正在进行的不法侵害，而采取的制止不法侵害的行为。正当防卫造成损害的，不负刑事责任。但正当防卫明显超过必要限度造成重大损害的，应当负刑事责任，但是应当减轻或者免除处罚。对正在进行行凶、杀人、抢劫、强奸、绑架以及其他严重危及人身安全的暴力犯罪，采取防卫行为，造成不法侵害人伤亡的，不属于防卫过当，不负刑事责任。"}]}
{"messages": [{"role": "user", "content": "离婚时财产如何分割？"}, {"role": "assistant", "content": "根据《民法典》第一千零八十七条，离婚时夫妻共同财产由双方协议处理；协议不成的，由人民法院根据财产的具体情况，按照照顾子女、女方和无过错方权益的原则判决。夫妻共同财产包括婚姻关系存续期间的工资收入、投资收益、知识产权收益等。个人财产（如婚前财产、因身体受到伤害获得的赔偿等）不属于分割范围。债务也需区分共同债务和个人债务分别处理。"}]}
```

> ⚠️ **注意**：以上数据仅作演示，实际法律模型需要由专业律师审核数据准确性。

### A.2 数据格式转换脚本

将任意 JSONL 文件转为训练框架可用格式：

```python
import json
from datasets import Dataset, DatasetDict

def prepare_dataset(jsonl_path: str, split_ratio: float = 0.9):
    """加载 ChatML 格式的 JSONL 并划分训练/验证集"""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    dataset = Dataset.from_list(data)
    split = dataset.train_test_split(test_size=1 - split_ratio, seed=42)
    return DatasetDict({"train": split["train"], "validation": split["test"]})

dataset = prepare_dataset("legal_data.jsonl")
print(dataset)
# DatasetDict({
#     train: Dataset({ features: ['messages'], num_rows: 9 })
#     validation: Dataset({ features: ['messages'], num_rows: 1 })
# })
```

### A.3 训练配置

#### 方案 A：使用 LLaMA-Factory（推荐，YAML 配置）

```yaml
# legal_qlora.yaml
model_name_or_path: Qwen/Qwen2.5-7B-Instruct
dataset: legal_data.jsonl
dataset_format: chatml
template: qwen

# QLoRA 配置
quantization_bit: 4
quantization_type: nf4
double_quantization: true

# LoRA 配置
lora_rank: 8
lora_alpha: 32
lora_dropout: 0.1
lora_target_modules: all

# 训练参数
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 2e-4
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.03
bf16: true

# 输出
output_dir: ./legal-qlora-checkpoints
logging_steps: 10
save_strategy: epoch
```

训练命令：

```bash
llamafactory-cli train legal_qlora.yaml
```

#### 方案 B：使用 HuggingFace TRL（Python 脚本）

```python
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset

# ---------- 4-bit 量化 ----------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# ---------- 模型 ----------
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model)

# ---------- LoRA ----------
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# ---------- Tokenizer ----------
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ---------- 数据 ----------
dataset = load_dataset("json", data_files="legal_data.jsonl", split="train")

def format_chatml(example):
    """将 ChatML 格式拼接为模型要求的模板"""
    text = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )
    return {"text": text}

dataset = dataset.map(format_chatml)

# ---------- 训练 ----------
training_args = TrainingArguments(
    output_dir="./legal-qlora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    max_seq_length=2048,
    dataset_text_field="text",
)

trainer.train()
```

### A.4 合并 LoRA 并导出

```python
from peft import PeftModel
import torch

# 加载基座模型（FP16，非量化）
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# 加载 LoRA 适配器
model = PeftModel.from_pretrained(base_model, "./legal-qlora/checkpoint-xxx")

# 合并权重
merged_model = model.merge_and_unload()

# 保存完整模型
merged_model.save_pretrained("./legal-merged")
tokenizer.save_pretrained("./legal-merged")
```

### A.5 使用 vLLM 部署

```python
from vllm import LLM, SamplingParams

# 加载合并后的模型
llm = LLM(model="./legal-merged", dtype="bfloat16")

# 推理
prompt = "请问借条和欠条有什么区别？"
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512,
)

# 使用 ChatML 格式
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("./legal-merged")
messages = [{"role": "user", "content": prompt}]
formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)

outputs = llm.generate([formatted_prompt], sampling_params)
print(outputs[0].outputs[0].text)
```

### A.6 评估

#### 人工评估

设计 10-20 个法律问答场景，人工检查回答的事实正确性和完整性：

```python
test_cases = [
    "试用期最长是多久？",
    "离婚财产如何分割？",
    "什么是正当防卫？",
    "公司不交社保违法吗？",
]
# 逐条人工评估
```

#### GPT-4-as-Judge 评估

```python
import openai

def evaluate_legal_answer(question: str, model_answer: str) -> dict:
    """用 GPT-4 评估法律问答的质量"""
    judge_prompt = f"""
    你是一个法律评估专家。请评估以下法律问答回答的质量。

    问题：{question}

    回答：{model_answer}

    请从以下维度评分（1-10分）：
    1. 事实准确性：回答中的法律条文和观点是否正确？
    2. 完整性：是否覆盖了问题的关键方面？
    3. 清晰度：表达是否清晰、通俗易懂？

    请以 JSON 格式输出评分和简要理由。
    """

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": judge_prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)
```

> 💡 **完整工作流总结**：数据准备（10 条起步）→ 格式转换 → QLoRA 训练（单卡 4090 约 10 分钟）→ LoRA 合并 → 导出 → vLLM 部署 → 多维度评估。整个过程在一张消费级 GPU 上即可完成，总成本不超过 1 美元（电费）。