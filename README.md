# COT-Valve

# 论文总结与实验细节

## 1. 论文总结

这篇论文主要解决链式思考（Chain-of-Thought, CoT）虽然能提升大模型推理能力，但其生成的长链条会显著增加推理成本的问题。

### 核心思想与贡献：

- **提出 CoT-Valve 策略**：[source: 3] 作者观察到，模型在处理简单任务时推理路径容易压缩，但在处理困难任务时则不然 [source: 2]。因此，他们提出了一种名为 CoT-Valve 的新的微调和推理策略，旨在让单个模型能够生成不同长度的推理链，从而根据任务难度动态调整推理开销 [source: 3, 45]。
- **参数空间方向控制**：[source: 4, 33] 实现 CoT-Valve 的关键在于，在模型的参数空间中找到一个特定的“方向”（表示为 Δθ）。通过调整沿这个方向更新的幅度（幅度因子 α），可以有效控制生成 CoT 的长度 [source: 4, 79-80]。增大步长（α > 1，外插）倾向于生成更短的链，减小步长（0 < α < 1，内插）则生成较长的链 [source: 83-86]。
- **利用 LoRA 实现**：[source: 35, 77] 作者选择使用 LoRA（Low-Rank Adaptation）来实现这个方向 Δθ 的控制。LoRA 作为一个额外的分支，参数量小，易于调整其影响强度（通过调整 α）。
- **构建 MixChain 数据集**：[source: 6, 46, 92-96] 为了训练和优化 CoT-Valve，作者利用该方法生成了 MixChain 数据集。该数据集的特点是，针对同一个问题，包含从长到短多个不同长度但都导向正确答案的推理链 [source: 38, 94]。
- **提出增强策略**：[source: 6, 47] 基于 MixChain 数据集，作者提出了两种增强策略：
  - **CoT-Valve++**：更精确的长度可压缩 CoT 微调方法，在训练中显式加入长度因子 β，使模型在训练时就学习适应不同长度 [source: 102-107]。
  - **CoT-Valve+P**：渐进式链长压缩方法，在训练时逐步使用 MixChain 中越来越短的推理链进行微调 [source: 108-111]。
- **实验验证**：[source: 7, 8, 41, 48] 实验证明 CoT-Valve 成功实现了 CoT 长度的可控性和可压缩性，并且在压缩 CoT 方面效果优于基于提示（Prompt-based）的方法。例如，在 QwQ-32B-Preview 模型上，GSM8K 数据集的推理链从 741 减少到 225 tokens，性能仅轻微下降 (95.07% -> 94.92%) [source: 8]。

---

## 2. 详细实验步骤

论文中的实验流程大致可以分为以下几个阶段（参考 Figure 2 [source: 81] 和 Section 3 [source: 61-111]、Section 4.1 [source: 112-119]）：

### 阶段 1：确定初始方向 Δθ

- **目标**：找到一个参数更新方向 Δθ，这个方向代表了从生成“长CoT”到生成“短CoT”的转变（或者反之，从无CoT到有CoT）。
- **方法**：
  1. **已有长短 CoT 数据 (Cold Start - Figure 2 左上)**：
     - 如果有一个包含同一问题长短两种 CoT 解决方案的数据集（例如，长 CoT 来自原始模型，短 CoT 来自人工标注或简化），可以微调模型分别学习这两种 CoT，得到两个模型参数 θ_long 和 θ_short。那么 Δθ 可以近似为 θ_short - θ_long [source: 97, 72]。
     - 论文中提到使用 GSM8K 或 PRM800k 作为这种场景的例子 [source: 97]。
  2. **利用现有模型对 (Zero-shot - Figure 2 右上)**：
     - 如果没有现成的长短 CoT 对照数据，可以利用一对现有的模型：一个基础 LLM (θ₁) 和一个基于它训练出来的、擅长长 CoT 的推理模型 (θ₂)。例如，基础模型是 LLaMA-3.1-8B (θ₁)，推理模型是 DeepSeek-R1-Distill-Llama-8B (θ₂) [source: 100, 128]。
     - 此时，Δθ = θ₂ - θ₁，代表了从基础模型到推理模型的“任务向量” [source: 73, 101]。

---

### 阶段 2：生成 MixChain 数据集

- **目标**：利用阶段 1 得到的 Δθ 来为训练集中的每个问题生成多个不同长度的 CoT 解决方案。
- **方法**：
  1. 选择一个基础模型参数 θ (通常是 θ₁ 或 θ₂ 中的一个，取决于阶段 1 的方法)。
  2. 选择一系列的插值/外插因子 α (例如论文中提到的 [0.2, 0.4, 0.6, 0.8] [source: 294] 或 α=1 等)。
  3. 对于训练集中的每个问题 q，使用调整后的模型参数 θ' = θ + αΔθ 来生成 CoT 解答 [source: 95]。
  4. 对每个 α 值都生成一次解答，这样每个问题 q 就对应了多个不同长度（可能）的 CoT 解答 {solution_α}。
  5. 过滤掉所有最终答案错误的解答 [source: 129]。
  6. 将这些 (question, answer, solution_α) 对组合起来，形成 MixChain 数据集（MixChain-C 或 MixChain-Z 取决于阶段 1 的方法）[source: 93, 127, 128]。

---

### 阶段 3：使用 MixChain 进行增强训练 (可选)

- **目标**：利用 MixChain 数据集进一步优化模型，以获得更精确的长度控制能力或更好的压缩效果。
- **方法**：
  1. **CoT-Valve++ (精确控制)** [source: 102-107]:
     - 为 MixChain 中的每个 solution 计算一个归一化长度因子 β (公式 4 [source: 106])，表示其相对长度 (β=0 对应最长，β=1 对应最短)。
     - 训练一个新的 LoRA 模块 Δθ'。训练目标是让模型 θ + βΔθ' 能够准确预测对应 β 长度的 solution (公式 3 [source: 105])。
  2. **CoT-Valve+P (渐进压缩)** [source: 108-111, 185]:
     - 将 MixChain 中的 solutions 按长度排序（从长到短）。
     - 分阶段进行训练。例如，先用最长的 solution (如 solution_4) 训练模型几轮，然后用次长的 (solution_3) 继续训练，以此类推，最后用最短的 solution (solution_0 或 solution_1) 训练。论文中提到每个阶段训练 2 个 epoch [source: 274, 280]。

---

### 阶段 4：评估

- **模型**：论文中测试了多种模型，包括 QwQ-32B-Preview, DeepSeek-R1-Distill-Llama-8B, LLaMA-3.1-8B, LLaMA-3.2-1B, Qwen-32B-Instruct [source: 112]。
- **数据集**：主要使用 GSM8K [source: 204] 和 AIME [source: 124] 进行评估。
- **指标**：Accuracy (准确率), #Tokens (生成答案的平均 Token 数), ACU (Accuracy per Computation Unit，综合考虑准确率、参数量和 Token 数的效率指标) [source: 116-118]。
- **推理**：在推理时，通过调整 LoRA 权重中的 α 因子来控制生成 CoT 的长度 [source: 86]。例如，加载基础模型和训练好的 LoRA 权重 (Δθ 或 Δθ')，然后以 θ_base + α * Δθ 的形式进行推理。


我要评估的目标数据集是：
load_dataset("openai/gsm8k","main")['test']
load_dataset("HuggingFaceH4/MATH-500")['test']
还有https://github.com/chaochun/nlu-asdiv-dataset

---

## 3. 复现代码结构 (概念性)

由于论文没有提供官方代码库，这里提供一个基于 Hugging Face transformers, peft (用于 LoRA) 和 datasets 库的概念性代码结构。你需要根据论文细节和你的具体环境进行填充和修改。

## 一些可能有用的原文：

Introduction
We propose a new tuning and inference strategy named CoT-Valve, designed to allow models to generate reasoning chains of varying lengths.

We propose to identify a direction in the parameter space that, when manipulated, can effectively control the length of generated CoT.
We construct datasets with chains from long to short for the same questions and explore two enhanced strategies for CoT-Valve: (1) a precise length-compressible CoT tuning method, and (2) a progressive chain length compression approach.
CoT-Valve successfully enables controllability and compressibility of the chain and shows better performance than the prompt-based control.
We applied this method to QwQ-32B-Preview, reducing reasoning chains on GSM8K from 741 to 225 tokens with a minor performance drop (95.07% to 94.92%) and on AIME from 6827 to 4629 tokens, with only one additional incorrect answer.


(Long to Short CoT) For QwQ-32B-Preview (QwQ for abbreviation) and DeepSeek-R1Distill-Llama-8B (R1-Distill), we used our method to control and compress the length of the reasoning chain.

Training and Evaluation. For training the model, we use LoRA (Hu et al., 2022) in most of our experiments, except in the experiment for LIMO on Qwen-2.5-32B-Instruct we use full parameter fine-tuning.

Dataset Explanation  As detailed in Section 4.2, we constructed two types of datasets: MixChain-C and MixChain-Z. The statistics for the datasets are shown in 9. For these datasets, we select α values ranging from [0.6, 0.8] for LIMO and [0.2, 0.4, 0.6, 0.8] for other datasets, ensuring all incorrect responses are excluded. For MixChain-Z, while the training transition from θ1 to θ2 remains a black box, we can still identify numerous model pairs such as Qwen-32BInstruct → QwQ-32B-Preview, and LLaMA-3.18B → R1-Distill-Llama-8B, as documented in the technical report. We find that the performance of the base model significantly influences the quality of the dataset.

Controllable Results. We illustrate the result in Figure 3a. First, using ground-truth samples as a cold start, we develop a model capable of generating reasoning paths of various lengths, as demonstrated in ‘CoT-Valve’ in Figure 3a. CoT-Valve already matches the performance of prompt-based control but can generate shorter reasoning chains. We then extrapolate ∆θ to produce even shorter reasoning paths. Then, building on MixChain-C from this first model, we conduct further training by CoTValve++. CoT-Valve++ substantially surpasses the baseline and shows greater generalization capabilities in cases of extrapolation.
