# References：调研参考文献与资源

> 本文档汇总 Project2_TaskB_Survey 所引用的代表性论文、开源项目和评估基准，按主题分类整理。
>
> 所有引用均来源于三份技术笔记和项目已有文档。仅列出核心文献，完整引用列表请参见各笔记的参考资源章节。

---

## 一、Alignment 方法

### 核心论文

1. **InstructGPT / RLHF**
   - Ouyang, L., et al. (2022). *Training language models to follow instructions with human feedback.* NeurIPS 2022.
   - [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
   - 提出了基于人类反馈的强化学习（RLHF）三阶段管线：SFT → Reward Model → PPO，ChatGPT 的方法论基础。

2. **DPO（Direct Preference Optimization）**
   - Rafailov, R., et al. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* NeurIPS 2023.
   - [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
   - 消去 Reward Model 和 RL，直接在偏好数据上用分类损失优化——将 Alignment 门槛大幅降低。

3. **RLAIF**
   - Lee, H., et al. (2023). *RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback.*
   - [arXiv:2309.00267](https://arxiv.org/abs/2309.00267)
   - 用 AI 模型替代人类标注偏好数据，解决 RLHF 的可扩展性瓶颈。

4. **Constitutional AI**
   - Bai, Y., et al. (2022). *Constitutional AI: Harmlessness from AI Feedback.*
   - [arXiv:2212.08073](https://arxiv.org/abs/2212.08073)
   - Anthropic 提出：用一套自然语言的"宪法"原则指导 AI 自我修正，两阶段流程（监督修正 + RL 优化）。

5. **Anthropic HHH**
   - Askell, A., et al. (2021). *A General Language Assistant as a Laboratory for Alignment.*
   - [arXiv:2112.00861](https://arxiv.org/abs/2112.00861)
   - 提出 HHH（Helpful, Honest, Harmless）对齐框架，定义了 Alignment 的核心维度和张力关系。

6. **LIMA**
   - Zhou, C., et al. (2023). *LIMA: Less Is More for Alignment.*
   - [arXiv:2305.11206](https://arxiv.org/abs/2305.11206)
   - 仅用 1,000 条精心挑选的样本做 SFT，效果接近 RLHF——验证数据质量压倒数据数量。

### DPO 变体

7. **ORPO**
   - Hong, J., et al. (2024). *ORPO: Monolithic Preference Optimization without Reference Model.*
   - [arXiv:2403.07691](https://arxiv.org/abs/2403.07691)
   - 将 SFT 和偏好优化合并为单阶段训练，无需参考模型。

8. **SimPO**
   - Meng, Y., et al. (2024). *SimPO: Simple Preference Optimization with a Reference-Free Reward.*
   - [arXiv:2405.14734](https://arxiv.org/abs/2405.14734)
   - 用序列平均对数概率作为隐式奖励，长度归一化，无参考模型。

### 推理对齐

9. **DeepSeek-R1 / GRPO**
   - DeepSeek-AI. (2025). *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.*
   - [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
   - 用 GRPO（组相对策略优化）在可验证奖励上做 RL，不依赖人类偏好。AIME 2024 从 15.6% 提升至 71.0%。

---

## 二、Fine-tuning 方法

### 参数高效微调（PEFT）

10. **LoRA（Low-Rank Adaptation）**
    - Hu, E. J., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
    - [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
    - 用低秩矩阵 $BA$ 参数化权重更新，冻结预训练权重。可训练参数仅 0.1-1%，推理零额外开销。

11. **QLoRA**
    - Dettmers, T., et al. (2023). *QLoRA: Efficient Finetuning of Quantized Language Models.* NeurIPS 2023.
    - [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
    - NF4 量化 + 双重量化 + 分页优化器，使 65B 模型可在单卡 RTX 3090（24GB）上微调。

12. **Adapter**
    - Houlsby, N., et al. (2019). *Parameter-Efficient Transfer Learning for NLP.* ICML 2019.
    - [arXiv:1902.00751](https://arxiv.org/abs/1902.00751)
    - 在 Transformer 层间插入瓶颈网络，最早的主流 PEFT 方法。

13. **Prefix Tuning**
    - Li, X. L. & Liang, P. (2021). *Prefix-Tuning: Optimizing Continuous Prompts for Generation.* ACL 2021.
    - [arXiv:2101.00190](https://arxiv.org/abs/2101.00190)
    - 在每层 Attention 的 K/V 前添加可训练的连续前缀向量。

14. **Prompt Tuning**
    - Lester, B., et al. (2021). *The Power of Scale for Parameter-Efficient Prompt Tuning.* EMNLP 2021.
    - [arXiv:2104.08691](https://arxiv.org/abs/2104.08691)
    - 仅在输入层添加可学习的 soft prompt embedding，参数最少（~几 KB）。

### LoRA 改进变体

15. **DoRA**
    - Liu, S., et al. (2024). *DoRA: Weight-Decomposed Low-Rank Adaptation.* ICML 2024.
    - [arXiv:2402.09353](https://arxiv.org/abs/2402.09353)
    - 将预训练权重分解为幅度和方向，幅度直接学习，方向用 LoRA，常优于 LoRA 1-3%。

16. **AdaLoRA**
    - Zhang, Q., et al. (2023). *AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning.* ICLR 2023.
    - [arXiv:2303.10512](https://arxiv.org/abs/2303.10512)
    - 根据重要性分数动态分配秩，重要模块大 $r$，不重要模块小 $r$。

17. **LoRA+**
    - Hayou, S., et al. (2024). *LoRA+: Efficient Low-Rank Adaptation of Large Models.*
    - [arXiv:2402.12354](https://arxiv.org/abs/2402.12354)
    - 为 $A$ 和 $B$ 矩阵设置不同学习率（$\eta_B = 2\eta_A$），收敛速度提升约 2 倍。

18. **PiSSA**
    - Meng, F., et al. (2024). *PiSSA: Principal Singular Values and Singular Vectors Adaptation.*
    - [arXiv:2404.02948](https://arxiv.org/abs/2404.02948)
    - 用预训练权重的 SVD 主成分初始化 LoRA 的 $A,B$，收敛更快，少数据场景优于 LoRA。

19. **VeRA**
    - Kopiczko, D. J., et al. (2023). *VeRA: Vector-based Random Matrix Adaptation.*
    - [arXiv:2310.11454](https://arxiv.org/abs/2310.11454)
    - 共享随机矩阵 + 仅学缩放向量，参数量仅为 LoRA 的 1/10。

### 指令微调与数据合成

20. **Alpaca**
    - Taori, R., et al. (2023). *Alpaca: A Strong, Replicable Instruction-Following Model.* Stanford CRFM.
    - [https://crfm.stanford.edu/2023/03/13/alpaca.html](https://crfm.stanford.edu/2023/03/13/alpaca.html)
    - 用 Self-Instruct + text-davinci-003 生成 52K 指令数据，微调 LLaMA 7B，总成本 ~$100。

21. **Self-Instruct**
    - Wang, Y., et al. (2022). *Self-Instruct: Aligning Language Models with Self-Generated Instructions.* ACL 2023.
    - [arXiv:2212.10560](https://arxiv.org/abs/2212.10560)
    - 用种子指令 + 强模型扩展生成指令数据的方法论基础，Alpaca 的核心数据管道。

22. **LLaMA**
    - Touvron, H., et al. (2023). *LLaMA: Open and Efficient Foundation Language Models.*
    - [arXiv:2302.13971](https://arxiv.org/abs/2302.13971)
    - Meta 的开源基座模型系列，Alpaca、Vicuna 等众多指令微调工作的基座。

---

## 三、预训练基础模型

23. **BERT**
    - Devlin, J., et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL 2019.
    - [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
    - 提出 Masked LM 预训练范式，"预训练 + 微调"模式的奠基之作。

24. **GPT-3**
    - Brown, T. B., et al. (2020). *Language Models are Few-Shot Learners.* NeurIPS 2020.
    - [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)
    - 展示了大规模自回归语言模型的涌现能力和 In-Context Learning。

25. **T5**
    - Raffel, C., et al. (2020). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.* JMLR 2020.
    - [arXiv:1910.10683](https://arxiv.org/abs/1910.10683)
    - 将所有 NLP 任务统一为"文本到文本"格式，前缀语言模型代表。

---

## 四、评估框架与基准

### 评估框架

26. **EleutherAI lm-evaluation-harness**
    - Gao, L., et al. (2023). *A Framework for Few-Shot Language Model Evaluation.*
    - [GitHub](https://github.com/EleutherAI/lm-evaluation-harness) | [论文](https://arxiv.org/abs/2305.18290)
    - 最流行的开源 LLM 评估框架，200+ 预定义任务，被 HuggingFace Open LLM Leaderboard 采用。

27. **HELM（Holistic Evaluation of Language Models）**
    - Liang, P., et al. (2022). *Holistic Evaluation of Language Models.*
    - [arXiv:2211.09110](https://arxiv.org/abs/2211.09110)
    - Stanford CRFM 的全面评估框架，覆盖 7 个维度（准确性、校准度、鲁棒性、公平性等）。

28. **BIG-Bench**
    - Srivastava, A., et al. (2022). *Beyond the Imitation Game: Quantifying and Extrapolating the Capabilities of Language Models.*
    - [arXiv:2206.04615](https://arxiv.org/abs/2206.04615)
    - 200+ 任务的大规模协作评估基准，覆盖推理、知识、创造力等多维度。

### 核心 Benchmark

29. **MMLU**
    - Hendrycks, D., et al. (2020). *Measuring Massive Multitask Language Understanding.* ICLR 2021.
    - [arXiv:2009.03300](https://arxiv.org/abs/2009.03300)
    - 57 学科、~15,908 题的多项选择基准，LLM 知识广度的最常用度量。

30. **GSM8K**
    - Cobbe, K., et al. (2021). *Training Verifiers to Solve Math Word Problems.*
    - [arXiv:2110.14168](https://arxiv.org/abs/2110.14168)
    - 8,500 道小学数学应用题，需要多步推理，当前评估数学推理能力的标准基准。

31. **HumanEval**
    - Chen, M., et al. (2021). *Evaluating Large Language Models Trained on Code.*
    - [arXiv:2107.03374](https://arxiv.org/abs/2107.03374)
    - 164 道编程题，Pass@k 指标，通过运行测试用例判断功能正确性。

32. **TruthfulQA**
    - Lin, S., et al. (2022). *TruthfulQA: Measuring How Models Mimic Human Falsehoods.* ACL 2022.
    - [arXiv:2109.07958](https://arxiv.org/abs/2109.07958)
    - 817 道对抗性设计的问题，测量模型是否重复人类常见误解——诚实性的核心基准。

33. **HellaSwag**
    - Zellers, R., et al. (2019). *HellaSwag: Can a Machine Really Finish Your Sentence?* ACL 2019.
    - [arXiv:1905.07830](https://arxiv.org/abs/1905.07830)
    - ~10,000 道句子补全题，需要常识推理，随机基线 25%。

### 新基准（2024-2025）

34. **MMLU-Pro**
    - Wang, Y., et al. (2024). *MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark.*
    - [arXiv:2406.01574](https://arxiv.org/abs/2406.01574)
    - 剔除 MMLU 中的简单题，增加难度和选项数（4 → 10），降低 ceiling effect。

35. **LiveBench**
    - Li, J., et al. (2024). *LiveBench: A Dynamic Benchmark for Language Model Evaluation.*
    - [GitHub](https://github.com/LiveBench/LiveBench)
    - 每月从新闻、论文等最新来源更新数据，从根源上防止 Benchmark 污染。

36. **SWE-bench**
    - Jimenez, C. E., et al. (2024). *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?*
    - [arXiv:2310.06770](https://arxiv.org/abs/2310.06770)
    - 2,294 个真实 GitHub Issue，模型需在 Docker 环境中完成代码修改并运行测试——Agent 评估的新范式。

37. **GPQA**
    - Rein, D., et al. (2023). *GPQA: A Graduate-Level Google-Proof Q&A Benchmark.*
    - [arXiv:2311.12022](https://arxiv.org/abs/2311.12022)
    - 448 道研究生级专业问题（物理/化学/生物），专家级难度，当前最高难度问答基准。

38. **AlpacaEval 2.0**
    - Dubois, Y., et al. (2024). *AlpacaEval 2.0: An Automatic Evaluator for Instruction-Following Models.*
    - [GitHub](https://github.com/tatsu-lab/alpaca_eval)
    - 805 条指令，基于 GPT-4 Turbo 的对比胜率评估，快速评估对齐效果。

39. **MT-Bench**
    - Zheng, L., et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.*
    - [arXiv:2306.05685](https://arxiv.org/abs/2306.05685)
    - 80 个多轮对话主题，GPT-4 多维度评分（帮助性、准确性、相关性等）。

40. **SafeRLHF / HarmBench**
    - PKU-Alignment (2023). *SafeRLHF: Safe Reinforcement Learning from Human Feedback.*
    - [GitHub](https://github.com/PKU-Alignment/safe-rlhf)
    - Mazeika, M., et al. (2024). *HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal.*
    - [arXiv:2402.04249](https://arxiv.org/abs/2402.04249)
    - 安全对齐的专项评估基准，覆盖有害内容拒绝率的对抗性测试。

---

## 五、代表 Benchmark 对比一览

| 基准 | 年份 | 样本数 | 评估维度 | 评估模式 | 核心用途 |
|------|------|--------|---------|---------|---------|
| MMLU | 2020 | ~15,908 | 57 学科知识 | Log-likelihood | 通用知识广度 |
| GSM8K | 2021 | ~8,500 | 数学推理 | 生成式 | 多步推理能力 |
| HumanEval | 2021 | 164 | 代码生成 | 生成式（执行测试） | 功能正确性 |
| HellaSwag | 2019 | ~10,000 | 常识推理 | Log-likelihood | 常识理解 |
| TruthfulQA | 2022 | 817 | 诚实性 | Log-likelihood | 幻觉检测 |
| MMLU-Pro | 2024 | ~12,000 | 更难的 57 学科 | Log-likelihood | 消除天花板效应 |
| LiveBench | 2024 | 动态 | 通用（动态更新） | 混合 | 防数据污染 |
| SWE-bench | 2024 | 2,294 | Agent 代码修复 | 生成式（执行测试） | Agent 能力 |
| GPQA | 2023 | 448 | 研究生级专业问答 | 多项选择 | 极高难度 |
| MT-Bench | 2023 | 80 主题 | 多轮对话 | 生成式 + GPT-4 评分 | 对话质量 |
| AlpacaEval 2.0 | 2024 | 805 | 指令遵循 | GPT-4 对比胜率 | 对齐效果 |
| HarmBench | 2024 | 多模式 | 安全拒绝率 | 对抗性生成式 | 安全性 |

---

## 六、核心开源项目

| 项目 | 维护方 | 用途 | 链接 |
|------|-------|------|------|
| **lm-evaluation-harness** | EleutherAI | 200+ 标准化评估任务 | [GitHub](https://github.com/EleutherAI/lm-evaluation-harness) |
| **PEFT** | HuggingFace | LoRA / QLoRA / DoRA 等 PEFT 方法 | [GitHub](https://github.com/huggingface/peft) |
| **TRL** | HuggingFace | SFT / DPO / PPO Trainer | [GitHub](https://github.com/huggingface/trl) |
| **BitsAndBytes** | Tim Dettmers | 4-bit / 8-bit 量化 | [GitHub](https://github.com/TimDettmers/bitsandbytes) |
| **Transformers** | HuggingFace | 模型加载与训练框架 | [GitHub](https://github.com/huggingface/transformers) |
| **LLaMA-Factory** | hiyouga (BIT) | 一站式微调/对齐 Web 平台 | [GitHub](https://github.com/hiyouga/LLaMA-Factory) |
| **OpenRLHF** | OpenRLHF | 分布式 RLHF 训练框架 | [GitHub](https://github.com/OpenRLHF/OpenRLHF) |
| **veRL** | 字节跳动/Volcengine | 高性能 RLHF 框架 | [GitHub](https://github.com/volcengine/verl) |
| **DeepSeek-R1** | DeepSeek | GRPO / 推理对齐开源实现 | [GitHub](https://github.com/deepseek-ai) |
| **mergekit** | Arcee AI | 模型权重合并（TIES/DARE/Linear） | [GitHub](https://github.com/arcee-ai/mergekit) |
| **OpenCompass** | 上海 AI Lab | 大模型评测平台（支持多模态） | [GitHub](https://github.com/open-compass/opencompass) |

---

## 七、技术笔记（本系列）

| 笔记 | 核心内容 |
|------|---------|
| **《Evaluation Harness 技术深度解析》** | 为什么要系统评估、Benchmark/Task/Metric/Few-shot 定义、Harness 架构设计、评估结果深入解读、Benchmark 污染与可复现性分析 |
| **《Alignment 技术笔记》** | RLHF/DPO/GRPO/RLAIF/Constitutional AI 的完整数学推导、方法对比与选型指南、评估基准（MT-Bench/AlpacaEval/SafeRLHF） |
| **《Fine-tuning 技术笔记》** | Full FT/LoRA/QLoRA 方法解析、本征维度理论、PEFT 变体对比、数据格式与超参数调优、资源估算与部署方案 |
| **lm_eval_example_commands.md**（本目录） | lm-evaluation-harness 命令行参考，从安装到高级用法的完整示例 |
| **tool_usage_examples.md**（本目录） | 各工具的完整介绍与可运行代码示例 |

---

> **说明**：本文档仅列出调研中最核心的参考文献。完整引用请参见三份技术笔记各自的参考资源章节。
> 作者与日期信息以原始论文为准，arXiv 链接指向最新版本。
