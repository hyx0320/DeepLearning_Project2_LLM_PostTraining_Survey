# lm-evaluation-harness 命令示例与详解

> 本文档基于 EleutherAI lm-evaluation-harness（v0.4.x）撰写。
> 涵盖从基础评估到高级用法的完整命令示例，配合参数解释与预期输出解读。

---

## 目录

1. [安装与环境配置](#1-安装与环境配置)
2. [基础评估命令](#2-基础评估命令)
3. [多任务联合评估](#3-多任务联合评估)
4. [不同模型后端的评估](#4-不同模型后端的评估)
5. [Log-likelihood 模式详解](#5-log-likelihood-模式详解)
6. [生成式评估模式详解](#6-生成式评估模式详解)
7. [加速评估的技巧](#7-加速评估的技巧)
8. [多模型对比与结果聚合](#8-多模型对比与结果聚合)
9. [自定义评估任务](#9-自定义评估任务)
10. [局限性排查指南](#10-局限性排查指南)

---

## 1. 安装与环境配置

### 1.1 安装方式

**方式一：pip 安装（推荐，稳定版）**

```bash
pip install lm-eval
```

**方式二：源码安装（获取最新功能）**

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
pip install -e .
```

> 💡 **版本说明**：本文档以 **lm-eval v0.4.x** 为准。不同版本的 CLI 参数可能有细微差异（如 `--output_path` 的格式、`--model_args` 的参数分隔方式）。建议使用 `pip install lm-eval>=0.4.5`。如果遇到命令不兼容，运行 `lm_eval --help` 查看当前版本支持的所有参数。

### 1.2 验证安装

```bash
lm_eval --help
```

预期输出应显示所有可用的命令行参数列表。如果提示 `command not found`，请检查 Python 的 bin 目录是否在 PATH 中。

---

## 2. 基础评估命令

### 2.1 评估一个 Hugging Face 模型：最简单的例子

```bash
lm_eval \
  --model hf \
  --model_args pretrained=mistralai/Mistral-7B-v0.1,trust_remote_code=True \
  --tasks mmlu \
  --device cuda:0 \
  --batch_size 4 \
  --num_fewshot 5 \
  --output_path ./results/mistral-7b-mmlu.json \
  --seed 42 \
  --log_samples
```

#### 参数逐项解释

| 参数 | 值 | 含义 |
|------|-----|------|
| `--model` | `hf` | 模型类型为 Hugging Face Transformers。Harness 的模型适配层（Model API）会根据此参数选择对应的模型加载器 |
| `--model_args` | `pretrained=...,trust_remote_code=True` | 传给模型加载器的参数。`pretrained` 指定模型名称（从 Hugging Face Hub 拉取）或本地路径。`trust_remote_code=True` 允许执行模型仓库中的自定义代码（部分模型需要，如 Qwen、LLaMA-3） |
| `--tasks` | `mmlu` | 要评估的任务名称。可以是单个任务、逗号分隔的多个任务、或任务组的名称（如 `"mmlu,gsm8k,hellaswag"`） |
| `--device` | `cuda:0` | 指定运行设备。第一块 GPU 为 `cuda:0`，CPU 为 `cpu`，多GPU可使用 `cuda:0,1,2,3` |
| `--batch_size` | `4` | 批大小。**越大推理越快，但显存需求也越大**。可以设为 `auto`（自动检测最优 batch size） |
| `--num_fewshot` | `5` | 每个测试样本前的示范样本数。MMLU 的标准设置为 5-shot |
| `--output_path` | `./results/...` | 结果输出文件路径。支持 JSON、CSV 格式 |
| `--seed` | `42` | 随机种子。保证**实验结果的可复现性**——论文中发表的结果必须能通过相同 seed 复现 |
| `--log_samples` | （flag） | 输出每个样本的详细结果。**强烈推荐开启**——没有它，你将只看到聚合指标，无法做 bad case 分析和诊断 |

#### 预期输出

```
|  Task  |Version|    Metric    |Value |   |Stderr|
|--------|------:|-------------|-----:|---|-----:|
|  mmlu  |   2.0 |acc          |0.6245|±  |0.0039|
|        |       |acc_norm     |0.6245|±  |0.0039|
```

### 2.2 输出指标解读

#### `acc`（Accuracy——准确率）

$$\text{acc} = \frac{\text{正确回答数}}{\text{总问题数}}$$

对于 MMLU 的 57 个学科，0.6245 意味着模型平均答对了 62.45% 的问题。随机猜测的基准确率为 25%（4 选 1），所以 62% 表明模型确实学到了知识。

**与人类对比（MMLU 参考值）**：

| 模型/基准 | 得分 | 发表时间 |
|-----------|------|---------|
| 随机猜测 | 25.0% | — |
| GPT-3 175B | 43.9% | 2020 |
| Llama 2 7B | 45.3% | 2023 |
| Mistral 7B | 62.5% | 2023 |
| Llama 2 70B | 68.9% | 2023 |
| Llama 3 8B | 66.4% | 2024 |
| Llama 3 70B | 82.0% | 2024 |
| GPT-4 | 86.4% | 2023 |
| 人类毕业生（跨学科） | ~75% | — |
| 人类专家（单学科） | ~90% | — |

#### `acc_norm`（Length-normalized Accuracy——长度归一化准确率）

**为什么需要长度归一化？** 在 log-likelihood 评估中，更长的选项天然会有更低的 log-likelihood（因为负对数概率累加）：

$$\text{score}(a) = \sum_{t=1}^{|a|} \log P(a_t | \cdots)$$

如果选项 A 是 "yes"（3 tokens），选项 B 是 "no, I don't think so"（6 tokens），即使 B 是正确答案，它的总 log-likelihood 也可能更低——因为累加了更多的负值。

**归一化方式**：

$$\text{score}_{\text{norm}}(a) = \frac{1}{|a|} \sum_{t=1}^{|a|} \log P(a_t | \cdots) = \frac{\text{score}(a)}{|a|}$$

当所有选项长度相近时（如 MMLU 的 A/B/C/D 单个字母），`acc` 和 `acc_norm` 几乎相同。

#### `Stderr`（Standard Error——标准误差）

$$\text{Stderr} \approx \sqrt{\frac{\text{acc} \cdot (1 - \text{acc})}{N}}$$

对于 MMLU 的 $N = 14{,}042$（实际子集大小），$\text{acc} = 0.6245$：

$$\text{Stderr} \approx \sqrt{\frac{0.6245 \times 0.3755}{14042}} \approx \sqrt{\frac{0.2345}{14042}} \approx 0.00409$$

**解读**：真值有 95% 的概率落在 $\text{acc} \pm 1.96 \times \text{Stderr} \approx [0.6165, 0.6325]$ 区间内。这意味着：
- 模型 B 得分为 0.6300：与当前模型的差异在误差范围内——两者可能实力相当
- 模型 C 得分为 0.6400：超出误差区间——差异具有统计学意义

---

## 3. 多任务联合评估

### 3.1 同时评估多个 benchmark

```bash
lm_eval \
  --model hf \
  --model_args pretrained=Qwen/Qwen2.5-7B-Instruct \
  --tasks mmlu,gsm8k,hellaswag,arc_challenge,truthfulqa \
  --device cuda:0 \
  --batch_size 8 \
  --num_fewshot 5 \
  --output_path ./results/qwen2.5-7b-full.json \
  --seed 42
```

#### 预期输出

```
|           Task           |Version|    Metric    |Value |   |Stderr|
|--------------------------|------:|-------------|-----:|---|-----:|
|  mmlu                    |   2.0 |acc          |0.7050|±  |0.0040|
|  gsm8k                   |   3.0 |exact_match  |0.6450|±  |0.0130|
|  hellaswag               |   2.0 |acc_norm     |0.8100|±  |0.0039|
|  arc_challenge           |   2.0 |acc_norm     |0.6000|±  |0.0090|
|  truthfulqa              |   2.0 |acc          |0.5200|±  |0.0150|
```

### 3.2 多任务结果的综合诊断

单一 benchmark 的得分无法反映模型的全貌。只有跨多个维度的评估才能揭示模型的真实能力边界。

**诊断解读示例**（以上述 Qwen2.5-7B 的假想输出为例）：

| 任务 | 得分 | 诊断 |
|------|------|------|
| **MMLU 70.5%** | 知识覆盖良好，接近 7B 级别模型的较高水平 | ✅ 通用知识扎实 |
| **GSM8K 64.5%** | 数学推理能力不错但仍有提升空间（7B 模型通常 50-70%） | ⚠️ 推理能力中等 |
| **HellaSwag 81.0%** | 常识推理能力优秀（随机基线 25%，人类上限 ~95%） | ✅ 常识理解好 |
| **ARC-Challenge 60.0%** | 科学推理中等偏上（随机基线 25%，7B 模型通常 50-65%） | ✅ 科学推理可接受 |
| **TruthfulQA 52.0%** | Truthfulness 仅 52%，说明模型仍有较强的"自信地胡说"倾向，**幻觉问题严重**（随机基线 50%——因为题目设计为对抗常见误解） | ❌ 幻觉严重，需加强对齐 |

> **综合诊断**：该模型在常识推理（HellaSwag 81%）上表现优异，数学推理（GSM8K 64.5%）处于中等水平，但 truthfulness（TruthfulQA 52%）仅比随机基线好一点点——说明**模型知道很多，但不知道自己的知识边界**。这在实际部署中需要特别注意，可能需要额外做对齐（alignment）训练（如 DPO 或 RLHF 中的诚实性偏好优化）。关于对齐方法与评估的关联，详见《Alignment 技术笔记》第 8.4 节（对齐效果的评估基准）。

### 3.3 异常案例诊断

假设你在评估一个新微调后的模型时，发现以下异常结果：

```
| Task         | Metric | 当前值 | 基线值  | 差距   |
|------------- |--------|-------|--------|--------|
| MMLU         | acc    | 0.68  | 0.70   | -2%    |
| GSM8K        | em     | 0.45  | 0.60   | **-15%** |
| HellaSwag    | acc_norm | 0.82 | 0.80 | +2%    |
```

**异常点**：GSM8K 分数从基线的 60% 骤降至 45%，降幅远超其他任务（15% vs 2%）。这不是正常的"对齐税"——对齐税通常为 1-3%，而不是 15%。

**可能的原因分析**：

| 怀疑方向 | 验证方法 | 示例 |
|---------|---------|------|
| **数据污染**（微调数据与 GSM8K 测试集分布不符） | 检查微调数据中是否有数学推理样本的格式差异 | GSM8K 需要逐步推理，若微调数据只有答案没有推理链，模型会"忘记"如何逐步推理 |
| **数学能力遗忘**（Catastrophic Forgetting） | 与其他数学 benchmark（如 MATH、BBH）交叉验证 | 如果 MATH 也下降，说明数学能力整体退化；如果仅 GSM8K 下降，可能是格式不匹配 |
| **Tokenization 偏移** | 检查 GSM8K 中数字的 tokenize 方式在微调前后是否一致 | 某些 tokenizer 对多位数编码不一致（如 "123" → [1,2,3] vs [123]），导致概率估计异常 |
| **对齐税（Alignment Tax）** | 对比微调前后的 logits 分布 | RLHF/DPO 训练可能抑制了"过度推理"的行为，误伤数学能力 |

**结论**：单一任务异常时，不要直接归因于模型能力下降。需要结合交叉验证、输入检查和分维度分析来定位真正原因。这种系统性的诊断思维是评估工作的核心能力。

---

## 4. 不同模型后端的评估

lm-eval 通过模型适配层（Model API）支持多种模型后端：

```
模型适配层            支持的模型类型
├── hf                Hugging Face Transformers（本地或 Hub）
├── vllm              vLLM（PagedAttention 加速）
├── openai            OpenAI API（GPT-3.5, GPT-4 等）
├── openai-completions OpenAI Completions API
├── local-completions 本地部署的 API 兼容服务
├── anthropic         Anthropic API（Claude 系列）
├── gguf              GGUF 格式（llama.cpp 生态）
├── tensorrt-llm      NVIDIA TensorRT-LLM
└── ...               更多后端持续添加中
```

### 4.1 评估 Hugging Face 模型（最常用）

```bash
# 从 Hugging Face Hub 加载
lm_eval \
  --model hf \
  --model_args pretrained=meta-llama/Meta-Llama-3-8B,dtype=bfloat16 \
  --tasks hellaswag \
  --device cuda:0 \
  --batch_size auto \
  --num_fewshot 5

# 加载本地 checkpoint
lm_eval \
  --model hf \
  --model_args pretrained=./my_finetuned_model,trust_remote_code=True \
  --tasks gsm8k \
  --device cuda:0

# 加载 PEFT/LoRA 微调后的模型（需先合并权重）
# 推荐先用 merge_and_unload() 合并 LoRA，再评估合并后的模型
```

### 4.2 评估通过 vLLM 部署的模型（5-10× 加速）

```bash
# 启动 vLLM 服务（在另一个终端或后台）
# python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct --port 8000

# 通过 vLLM 后端评估
lm_eval \
  --model vllm \
  --model_args pretrained=Qwen/Qwen2.5-7B-Instruct,tensor_parallel_size=2,dtype=bfloat16,gpu_memory_utilization=0.85 \
  --tasks mmlu,gsm8k \
  --batch_size auto \
  --num_fewshot 5
```

**vLLM 的优势**：
- 使用 PagedAttention，显存利用率大幅提升
- 支持 Continuous Batching，吞吐量高
- 相比 HF 直接推理，**评估速度可提升 5-10 倍**（特别适合生成式评估）

### 4.3 评估模型 API（OpenAI / Anthropic）

```bash
# OpenAI 模型
lm_eval \
  --model openai \
  --model_args model=gpt-4-0613,num_concurrent=8 \
  --tasks hellaswag \
  --num_fewshot 5

# Anthropic Claude
lm_eval \
  --model anthropic \
  --model_args model=claude-3-sonnet-20240229,num_concurrent=4 \
  --tasks mmlu \
  --num_fewshot 5
```

**API 评估的核心挑战**：
- **成本**：GPT-4 评估 MMLU（~14K 题）可能需要数千次 API 调用
- **速率限制**：需要设置 `num_concurrent` 控制并发数，避免被限流
- **非确定性**：API 模型可能不暴露 seed 控制，影响复现性
- **无法使用 log-likelihood 模式**：大多数 API 只提供生成式接口

### 4.4 评估本地部署的 API 兼容服务

当模型通过 vLLM、TGI（Text Generation Inference）、或 Ollama 部署为 API 服务时：

```bash
# local-completions 模式（适用于 Completion API）
lm_eval \
  --model local-completions \
  --model_args model=my_model,base_url=http://localhost:8000/v1/completions,num_concurrent=8 \
  --tasks hellaswag \
  --num_fewshot 5

# local-chat-completions 模式（适用于 Chat Completion API，较新版本）
# 适用于 Qwen、LLaMA-3 Chat 等指令模型
lm_eval \
  --model local-chat-completions \
  --model_args model=Qwen2.5-7B-Instruct,base_url=http://localhost:8000/v1/chat/completions,num_concurrent=8 \
  --tasks mmlu \
  --num_fewshot 5
```

---

## 5. Log-likelihood 模式详解

### 5.1 原理

Log-likelihood（对数似然）评估是 lm-eval 中最默认、最有效的评估方式。它**不需要模型生成文本**，而是直接计算模型对给定候选项的概率。

**核心公式**：给定上下文 $c$ 和候选项 $a_1, a_2, \ldots, a_m$，计算每个候选项的对数似然：

$$\text{score}(a_i) = \sum_{t=1}^{|a_i|} \log P(a_i^{(t)} | c, a_i^{(<t)}; \theta)$$

选择得分最高的候选项作为模型输出。

### 5.2 适用任务

| 任务 | 为什么用 Log-likelihood | 形式 |
|------|------------------------|------|
| **MMLU** | 4 选 1，只需要比较 4 个字母或单词的概率 | 多项选择 |
| **HellaSwag** | 选择最合理的句子结尾 | 句子补全 |
| **ARC** | 科学多项选择 | 多项选择 |
| **TruthfulQA** | 判断真/假陈述 | 二选判断 |
| **PIQA** | 选择最合理的物理交互方式 | 多项选择 |

### 5.3 哪些任务使用 Log-likelihood

lm-eval 的 YAML 任务定义中，`output_type: multiple_choice` 或 `output_type: perplexity` 的任务使用 log-likelihood 评估。

可以通过以下命令查看任务的评估方式：

```bash
# 查看某个任务的详细定义（包括 output_type）
lm_eval --tasks list --filter mmlu
```

### 5.4 Log-likelihood 评估的命令示例

```bash
# 这些任务天然使用 log-likelihood 模式
lm_eval \
  --model hf \
  --model_args pretrained=mistralai/Mistral-7B-v0.1 \
  --tasks mmlu,hellaswag,arc_challenge,truthfulqa \
  --device cuda:0 \
  --num_fewshot 5
```

### 5.5 两种加速技巧用于 Log-likelihood

```bash
# 技巧 1：使用 vLLM 后端（虽然 log-likelihood 计算方式不同，但 vLLM 有专门优化）
lm_eval --model vllm --model_args pretrained=...,dtype=bfloat16 --tasks mmlu

# 技巧 2：启用缓存，避免重复 tokenize
lm_eval --model hf --model_args pretrained=... --tasks mmlu --cache_requests true
```

### 5.6 Log-likelihood 的优点与局限

**优点**：
- **确定性**：不涉及采样，结果完全由 seed 和模型权重决定
- **高效**：只需一次前向传播即可比较所有候选项
- **信噪比高**：直接反映模型对正确答案的"确信程度"

**局限**：
- **与真实使用不一致**：用户使用时是"生成答案"，而不是"从 4 个选项中挑选"
- **只能用于封闭集合任务**：无法评估开放生成的场景（如对话质量、写作、翻译）
- **选项偏差**：如果正确选项写得不好（语法错误、措辞不当），即使模型知道答案也可能选错

> **关键认知**：同一个模型在 log-likelihood 模式下 MMLU 得分 70%，但在实际对话中可能表现远差于此——因为真实场景是**生成式**的，模型需要自主产生回答，而不是从给定的选项中挑选。"能识别正确答案"和"能生成正确答案"是两种不同的能力。

---

## 6. 生成式评估模式详解

### 6.1 原理

生成式评估（generate-until）让模型自回归生成文本，直到遇到终止条件（如换行符、EOS token），然后与参考答案比较。

$$\hat{y} = \arg\max_{y: \text{len}(y) < L} P(y | x; \theta) \quad \text{(greedy decoding)}$$

或使用 beam search、nucleus sampling 等策略。

### 6.2 适用任务

| 任务 | 为什么用生成式 | 匹配方式 |
|------|---------------|---------|
| **GSM8K** | 模型需要逐步推理并给出最终答案 | 正则提取答案后 Exact Match |
| **MATH** | 复杂数学推理 | 正则提取答案后 Exact Match |
| **HumanEval** | 需生成完整可运行的代码 | 运行代码，检查是否通过测试用例 |
| **MT-Bench** | 开放对话质量评估 | GPT-4 评分（LLM-as-Judge） |
| **翻译任务** | 开放生成翻译文本 | BLEU、COMET 等指标 |

### 6.3 生成式评估的命令示例

```bash
# GSM8K 使用生成式评估
lm_eval \
  --model hf \
  --model_args pretrained=Qwen/Qwen2.5-7B-Instruct \
  --tasks gsm8k \
  --device cuda:0 \
  --num_fewshot 5 \
  --batch_size 8 \
  --output_path ./results/qwen-gsm8k.json

# 自定义生成参数（如控制 temperature）
# lm-eval v0.4.x 支持通过 model_args 传递 generation_kwargs
lm_eval \
  --model hf \
  --model_args pretrained=...,generation_kwargs={'temperature':0.7,'top_p':0.9,'max_new_tokens':512} \
  --tasks gsm8k
```

> ⚠️ **注意**：`generation_kwargs` 的具体传参方式因 lm-eval 版本和模型后端而异。如果上述方式不生效，请检查文档或查看任务 YAML 中的 `generation_kwargs` 配置。

### 6.4 生成结果后处理与匹配

生成式评估的难点在于**如何从模型输出的自由文本中准确提取答案**。以 GSM8K 为例：

```
模型输出（生成文本）：
  "首先，张三有 5 个苹果，李四给了他 3 个。\n所以 5 + 3 = 8。\n因此张三现在有 8 个苹果。\n答案是 8。"

答案提取（lm-eval 内置正则）：
  "8"

参考答案：
  "8"

匹配结果：✅ Exact Match
```

lm-eval 在处理 GSM8K 时，内置了针对性的正则表达式来提取最终答案（通常匹配 `####` 后的数字，或 `答案是` 后的内容）。

### 6.5 Log-likelihood vs 生成式评估对比

| 特性 | Log-likelihood | 生成式（Generate-until） |
|------|---------------|------------------------|
| **计算方式** | 单次前向传播 | 自回归逐步生成 |
| **速度** | 快 | 慢 10-100× |
| **确定性** | 完全确定 | 受采样策略影响（temperature, top_p） |
| **与真实使用匹配度** | 低 | 高 |
| **适用任务** | 多项选择、分类 | 数学推理、代码生成、开放问答 |
| **典型 Benchmark** | MMLU, ARC, HellaSwag, TruthfulQA | GSM8K, HumanEval, MT-Bench, MATH |
| **显存消耗** | 低 | 高（需存储 KV cache 逐步生成） |

### 6.6 混合评估模式

有些 benchmark 同时支持两种模式。例如，MMLU 默认使用 log-likelihood 评估，但也可以配置为生成式评估：

```yaml
# MMLU 的生成式版本需要自定义 YAML 任务配置
task: mmlu_gen
output_type: generate_until
# ... 其他配置
```

在实践中，**对于多项选择题，优先使用 log-likelihood 模式**（高效、确定）；对于开放生成场景（推理、代码、对话），必须使用生成式模式。

---

## 7. 加速评估的技巧

全量评估（如 MMLU 57 学科 + GSM8K + HellaSwag 等）可能耗时数天，尤其在模型规模大（70B+）或使用生成式评估时。以下优化技巧可大幅缩短评估周期：

| 技巧 | 加速效果 | 做法 |
|------|---------|------|
| **使用 vLLM 后端** | 5-10× | `--model vllm` — vLLM 的 PagedAttention 大幅提升吞吐量 |
| **启用缓存** | 避免重复计算 | `--cache_requests true` 复用 tokenized 数据 |
| **评估子集** | 按需加速 | `--tasks mmlu:physics` 仅评估物理子任务；先用少量样本验证代码 |
| **小模型调试** | — | 先用 7B 模型调通评估流程，再用目标模型（70B+）正式评估 |
| **并行 API 请求** | 线性加速 | `--model_args num_concurrent=10` 对 API 模型同时发起请求 |
| **自动 batch_size** | 优化吞吐 | `--batch_size auto` 让 harness 自动寻找最大 batch size |
| **使用更小的 batch** | 避免 OOM | 显存不足时降低 `--batch_size`，配合 `--model_args use_cache=True` |

### 7.1 使用 vLLM 后端的完整示例

```bash
lm_eval \
  --model vllm \
  --model_args pretrained=meta-llama/Meta-Llama-3-70B-Instruct,tensor_parallel_size=4,dtype=bfloat16,gpu_memory_utilization=0.9 \
  --tasks mmlu,gsm8k,hellaswag \
  --batch_size auto \
  --num_fewshot 5
```

`tensor_parallel_size=4` 意味着将模型分布到 4 张 GPU 上。

### 7.2 评估子集进行快速验证

```bash
# 只评估 MMLU 的物理学科子集（快速验证代码正确性）
lm_eval \
  --model hf \
  --model_args pretrained=Qwen/Qwen2.5-7B-Instruct \
  --tasks mmlu:physics \
  --num_fewshot 5 \
  --batch_size 4

# 只评估前 100 条 GSM8K 样本（快速验证生成式评估流程）
# 注意：lm-eval 没有直接的 --max_samples 参数，需要通过自定义 YAML 或数据集过滤实现
```

### 7.3 评测流水线脚本示例

建议将评估命令写成 Shell 脚本，保证完全复现：

```bash
#!/bin/bash
# eval_pipeline.sh - 标准化评估流水线

MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR="./results"
DATE=$(date +%Y%m%d)

# 记录环境信息
echo "lm-eval 版本: $(lm_eval --version 2>/dev/null || echo 'unknown')"
echo "PyTorch 版本: $(python -c 'import torch; print(torch.__version__)')"
echo "评估日期: $DATE"

# 评估一组任务
lm_eval \
  --model hf \
  --model_args pretrained=$MODEL_NAME,dtype=bfloat16,trust_remote_code=True \
  --tasks "mmlu,gsm8k,hellaswag,arc_challenge,truthfulqa" \
  --device cuda:0 \
  --batch_size auto \
  --num_fewshot 5 \
  --output_path "${OUTPUT_DIR}/${DATE}_${MODEL_NAME##*/}_full.json" \
  --seed 42 \
  --log_samples \
  --cache_requests true

echo "评估完成！结果保存至 ${OUTPUT_DIR}/${DATE}_${MODEL_NAME##*/}_full.json"
```

---

## 8. 多模型对比与结果聚合

实际工作中常需要对比多个模型在同组任务上的结果，以生成可分享的比较报告。

### 8.1 逐个评估再聚合

```bash
# 评估三个模型
models=("mistralai/Mistral-7B-v0.1" "meta-llama/Llama-2-7b-hf" "Qwen/Qwen2.5-7B")
tasks="mmlu,gsm8k,hellaswag,truthfulqa,arc_challenge"
num_fewshot=5

for model in "${models[@]}"; do
    model_name=$(echo $model | tr '/' '_')
    lm_eval \
        --model hf \
        --model_args pretrained=$model,dtype=bfloat16,trust_remote_code=True \
        --tasks $tasks \
        --device cuda:0 \
        --batch_size auto \
        --num_fewshot $num_fewshot \
        --output_path "./results/${model_name}_eval.json" \
        --seed 42
done
```

### 8.2 使用 Python 生成对比表与可视化

```python
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_results(model_paths: dict[str, str]) -> pd.DataFrame:
    """从多个模型的 JSON 结果文件中提取关键分数"""
    rows = []
    for model_name, path in model_paths.items():
        with open(path) as f:
            data = json.load(f)
        for task, metrics in data["results"].items():
            # 提取每个任务的标准指标
            for metric, value in metrics.items():
                if metric in ("acc", "acc_norm", "exact_match"):
                    rows.append({
                        "model": model_name,
                        "task": task,
                        "metric": metric,
                        "value": value * 100,  # 转为百分比
                    })
    return pd.DataFrame(rows)

# 使用示例
model_paths = {
    "Mistral-7B": "./results/mistralai_Mistral-7B-v0.1_eval.json",
    "Llama-2-7B": "./results/meta-llama_Llama-2-7b-hf_eval.json",
    "Qwen2.5-7B": "./results/Qwen_Qwen2.5-7B_eval.json",
}
df = load_results(model_paths)

# 生成 Markdown 对比表格
pivot_table = df.pivot_table(
    index="model",
    columns="task",
    values="value",
    aggfunc="mean"
).round(1)
print(pivot_table.to_markdown())

# 生成雷达图
def plot_radar(df: pd.DataFrame, save_path: str = "model_comparison.png"):
    """多模型多任务对比雷达图"""
    pivot = df.pivot_table(index="model", columns="task", values="value")
    categories = pivot.columns.tolist()
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for idx, model in enumerate(pivot.index):
        values = pivot.loc[model].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, "o-", label=model, linewidth=2)
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)
    ax.set_ylim(0, 100)
    ax.set_title("Model Comparison", size=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"雷达图已保存至 {save_path}")

plot_radar(df)
```

---

## 9. 自定义评估任务

lm-eval 支持通过 YAML 文件定义新的评估任务，无需修改框架代码。

### 9.1 自定义任务：一个完整的 YAML 示例

```yaml
# my_custom_task.yaml
task: my_custom_qa                     # 任务名称（唯一标识）
dataset_path: my_company/my_dataset    # HuggingFace 数据集路径或本地 JSONL
dataset_name: default                  # 数据集配置名（若数据集有子集）
test_split: test                       # 使用的数据划分
fewshot_split: train                   # few-shot 示例来源
num_fewshot: 3                         # few-shot 示例数量

# 输出类型：generate_until = 生成直到终止条件
output_type: generate_until

# 将数据行映射为 prompt
doc_to_text: "问题：{{question}}\n答案："

# 标准答案提取
doc_to_target: "{{answer}}"

# 生成终止条件
generation_kwargs:
  until:
    - "\n"
  max_gen_tokens: 128

# 评估指标
metric_list:
  - metric: exact_match               # 精确匹配
    aggregation: mean                  # 聚合方式（所有样本取均值）
    higher_is_better: true             # 越高越好
  - metric: f1                         # F1 分数
    aggregation: mean
    higher_is_better: true

# 可选的预处理（对模型输出做标准化后再比较）
process_results:
  - function: regex
    regex: "答案[：:]?\\s*([^\\n]+)"
    group: 1
```

### 9.2 使用自定义任务

```bash
# 使用 --tasks 指定 YAML 文件路径
lm_eval \
  --model hf \
  --model_args pretrained=Qwen/Qwen2.5-7B-Instruct \
  --tasks my_custom_task.yaml \
  --num_fewshot 3 \
  --device cuda:0 \
  --output_path ./results/custom_task.json
```

> 💡 YAML 任务文件可以放在任意路径，lm-eval 会自动加载。如果有多个自定义任务，建议放在 `lm_eval/tasks/` 目录下统一管理。更复杂的任务（如带动态 few-shot 选择、多轮交互）可能需要使用 Python 类定义。

---

## 10. 局限性排查指南

在使用 lm-eval 进行评估时，以下问题需要特别注意：

### 10.1 Benchmark 污染与检测

**定义**：预训练数据可能包含了 benchmark 数据集的测试集，导致模型不是在"推理"而是在"记忆"。

**数学表达**：假设模型在训练中见过测试样本 $(x, y^*)$，其条件概率不再反映泛化能力：

$$P(y | x; \theta) \to 1 \quad \text{不是因为推理，而是因为记忆}$$

**实际检出方法**：

```python
# n-gram 污染检测（核心思想）
def check_ngram_contamination(test_text: str, train_text: str, n: int = 13) -> bool:
    """检查测试样本是否与训练数据有 n-gram 重叠"""
    test_ngrams = set(zip(*[test_text.split()[i:] for i in range(n)]))
    train_ngrams = set(zip(*[train_text.split()[i:] for i in range(n)]))
    overlap = test_ngrams & train_ngrams
    contamination_rate = len(overlap) / len(test_ngrams)
    return contamination_rate > 0.5  # 超过 50% 的 n-gram 重叠视为污染

# 批量检查
for sample in test_dataset:
    for doc in train_corpus:
        if check_ngram_contamination(sample["text"], doc):
            print(f"WARNING: 样本 {sample['id']} 可能被污染")
```

**开源工具**：
- **GPT-NeoX contamination 检测脚本**：EleutherAI 提供的 n-gram 匹配工具
- **lm-eval 内置检测**：部分版本支持 `--check_integrity` 参数
- **DataPortrait**：专门的数据集污染分析工具

**针对污染的日常应对策略**：
- 优先使用新 benchmark（MMLU-Pro、LiveBench、Arena-Hard）
- 评估时关注**相对排名**而非绝对分数（所有模型都可能被污染）
- 使用动态更新的基准（如 LiveBench 每月更新数据）

### 10.2 可复现性保障

即使使用 evaluation harness，复现性仍然受以下因素影响：

| 因素 | 影响 | 解决方法 |
|------|------|---------|
| **GPU 精度** | FP16 vs BF16 vs FP32 导致微小但系统性的差异 | 在报告中注明精度设置 |
| **Transformers 库版本** | 不同版本模型加载差异 | 记录 `transformers.__version__` |
| **Flash Attention** | 启用/禁用可能改变 logits | 记录 Flash Attention 状态 |
| **Tokenizer 版本** | 更新可能改变编码方式 | 记录 tokenizer 版本 |
| **CUDA 版本** | 影响算子运行结果 | 记录 CUDA 版本 |

**最佳实践——每次评估都记录完整环境信息**：

```yaml
## 实验复现信息
评估目的: 对比 Qwen2.5-7B 与 Llama-3-8B 的通用能力
lm-eval 版本: v0.4.5
PyTorch 版本: 2.1.0
Transformers 版本: 4.36.2
CUDA 版本: 12.1
GPU 型号: A100-80GB
精度: bfloat16
Flash Attention: True
随机种子: 42
Batch Size: auto
任务列表: mmlu, gsm8k, hellaswag, arc_challenge, truthfulqa
Few-shot 数量: 5 (MMLU); 8 (GSM8K); 5 (HellaSwag); 0 (TruthfulQA)
评估日期: 2026-06-13
```

### 10.3 统计显著性检验

简单的均值比较可能产生误导。需要引入统计显著性检验：

**McNemar 检验**（配对样本，适合两个模型在相同数据上的对比）：

```python
from scipy.stats import chi2_contingency

def mcnemar_test(model_a_correct: list[bool], model_b_correct: list[bool]) -> float:
    """
    McNemar 检验：判断两个模型在配对样本上的表现是否有显著差异。
    返回 p-value，p < 0.05 表示差异显著。
    """
    # 构建 2x2 列联表
    n_01 = sum(1 for a, b in zip(model_a_correct, model_b_correct) if not a and b)      # A错B对
    n_10 = sum(1 for a, b in zip(model_a_correct, model_b_correct) if a and not b)      # A对B错

    table = [[0, n_01], [n_10, 0]]
    result = chi2_contingency(table, correction=True)
    return result.pvalue

# 示例
model_a_scores = [True, False, True, True, False, ...]  # 模型 A 每样本的正确/错误
model_b_scores = [True, True, True, False, False, ...]  # 模型 B
p_value = mcnemar_test(model_a_scores, model_b_scores)
print(f"McNemar p-value: {p_value:.4f}")
if p_value < 0.05:
    print("两个模型的表现存在显著差异")
else:
    print("差异未达到统计显著水平")
```

**Bootstrap 置信区间**：

```python
import numpy as np

def bootstrap_ci(scores: list[float], n_iterations: int = 10000, alpha: float = 0.05):
    """计算 Bootstrap 置信区间"""
    n = len(scores)
    boot_means = np.zeros(n_iterations)
    for i in range(n_iterations):
        sample = np.random.choice(scores, size=n, replace=True)
        boot_means[i] = np.mean(sample)
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return lower, upper

# 比较两个模型的置信区间是否重叠
ci_a = bootstrap_ci(model_a_accuracies)
ci_b = bootstrap_ci(model_b_accuracies)
print(f"模型 A 95% CI: [{ci_a[0]:.3f}, {ci_a[1]:.3f}]")
print(f"模型 B 95% CI: [{ci_b[0]:.3f}, {ci_b[1]:.3f}]")
if ci_a[1] < ci_b[0] or ci_b[1] < ci_a[0]:
    print("置信区间不重叠，差异显著")
else:
    print("置信区间重叠，差异可能不显著")
```

### 10.4 常见评估陷阱

| 陷阱 | 表现 | 解决办法 |
|------|------|---------|
| **辛普森悖论** | 总分对比正常，但所有子项都差 | 始终检查子任务级别的分数，不只看聚合值 |
| **版本不匹配** | 同一个 benchmark 在不同论文中分数不同 | 记录 lm-eval 的 `Version` 列（如 MMLU v2.0 vs v1.0） |
| **Prompt 格式敏感** | 不同 prompt 模板导致评分差异 5-15% | 使用与学术社区一致的 prompt 模板（lm-eval 已内置） |
| **Few-shot 选择偏差** | 不同示例导致结果大幅波动 | 固定 seed，报告多组 seed 的平均结果 |
| **对齐税误诊** | 把能力遗忘误认为对齐税 | 对齐税通常 < 3%；> 5% 的下降需要排查其他原因（数据污染、Tokenization 偏移等） |
| **数据污染** | benchmark 分数异常高 | 使用 LiveBench 等动态基准验证，或做 n-gram 污染检测 |

### 10.5 评估的完整性检查清单

- [ ] 是否在 **多个基准** 上评估（覆盖知识、推理、代码、安全等不同维度）？
- [ ] 是否记录了**完整的环境信息**（版本、精度、随机种子）？
- [ ] 是否评估了**生成式模式**（对于对话/推理类任务）？
- [ ] 是否与**基线模型**进行了对比（相同 seed、相同 prompt 模板）？
- [ ] 是否做了**统计显著性检验**？（差异是否超越 Stderr？）
- [ ] 是否检查了**子任务级别的分数**？（不仅仅是聚合后的总分）
- [ ] 是否排除了**数据污染**的可能？（对于异常高分）
- [ ] 是否在**多个 seed** 下重复了评估？（检验结果稳定性）
- [ ] 对于微调后的模型：是否评估了**通用能力的保持程度**？（防止遗忘）
- [ ] 对于对齐后的模型：是否评估了**安全性**（拒绝有害请求）和**诚实性**（TruthfulQA）？

---

## 参考资源

### 工具与文档

- **lm-evaluation-harness**: [https://github.com/EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- **Open LLM Leaderboard**: [https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)
- **OpenCompass**（上海 AI Lab）: [https://opencompass.org.cn](https://opencompass.org.cn)
- **HELM**（Stanford）: [https://crfm.stanford.edu/helm/latest/](https://crfm.stanford.edu/helm/latest/)

### 相关技术笔记（本系列）

- **《Evaluation Harness 技术深度解析》**：本文档的理论基础。涵盖为什么要系统评估、Benchmark/Task/Metric/Few-shot 的完整定义、Harness 的架构设计、评估结果深入解读方法（Calibration、辛普森悖论、统计检验）、以及 Benchmark 污染、可复现性危机等局限性分析。
- **《Alignment 技术笔记》** 第 8.4 节：对齐效果的评估基准（MT-Bench, AlpacaEval, SafeRLHF 等）。
- **《Alignment 技术笔记》** 第 8.5 节：多轮对齐的挑战——对话级 Reward、时序信用分配等。
- **《Fine-tuning 技术笔记》** 第 12 节：微调后的评估方法（LLM-as-Judge 偏差缓解、幻觉评估、多语言评估）。
- **《Fine-tuning 技术笔记》** 第 17 节：端到端管线（微调 → 对齐 → 评估）与常见失败模式诊断清单。