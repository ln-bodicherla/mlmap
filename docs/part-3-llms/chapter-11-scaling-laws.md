# Chapter 11: Scaling Laws and Emergent Abilities

> *"The most important thing in the programming field is the ability to recognize when a specific, clever technique has reached the point of diminishing returns."* -- Richard Sutton (paraphrased)

## Learning Objectives

By the end of this chapter, you will be able to:

1. State and interpret the neural scaling laws of Kaplan et al. (2020), including the power-law relationships between loss, model size, dataset size, and compute.
2. Explain how Chinchilla scaling (Hoffmann et al., 2022) revised the field's understanding of compute-optimal training and articulate its practical implications.
3. Critically evaluate claims of emergent capabilities, including the argument that emergence may be an artifact of evaluation metrics.
4. Describe inference-time compute scaling strategies and their relationship to training-time scaling.
5. Articulate the Bitter Lesson and its implications for AI research strategy.
6. Apply scaling laws to practical problems: estimating training costs, selecting model sizes, and planning training runs.

---

## 11.1 Neural Scaling Laws

One of the most consequential discoveries in modern AI is that language model performance follows remarkably predictable power-law relationships with scale. This was first systematically documented by Kaplan et al. (2020) at OpenAI and has since been refined, extended, and debated.

### 11.1.1 The Power-Law Relationships

Kaplan et al. (2020) observed that the cross-entropy loss $L$ of a language model follows smooth power-law relationships with three variables:

**1. Model size (number of non-embedding parameters $N$):**

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}$$

where $N_c \approx 8.8 \times 10^{13}$ and $\alpha_N \approx 0.076$.

**2. Dataset size (number of tokens $D$):**

$$L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}$$

where $D_c \approx 5.4 \times 10^{13}$ and $\alpha_D \approx 0.095$.

**3. Compute budget (FLOPs $C$):**

$$L(C) = \left(\frac{C_c}{C}\right)^{\alpha_C}$$

where $C_c \approx 3.1 \times 10^8$ and $\alpha_C \approx 0.050$.

These are empirical fits, but they hold over many orders of magnitude --- from models with millions of parameters to hundreds of billions, and from megabytes to terabytes of training data.

### 11.1.2 The Combined Scaling Law

When both model size and dataset size are varied simultaneously:

$$L(N, D) = \left[\left(\frac{N_c}{N}\right)^{\alpha_N / \alpha_D} + \frac{D_c}{D}\right]^{\alpha_D}$$

This functional form captures the interaction between model capacity and data: a larger model benefits less from additional data (it saturates faster), and more data benefits less when the model is too small to learn from it.

A key implication is that **performance improvements are predictable**. Given a model of size $N$ trained on $D$ tokens, one can estimate the loss with reasonable accuracy. This transforms model training from an empirical art into something closer to engineering.

### 11.1.3 What the Scaling Laws Tell Us

Several profound insights emerge:

**Power laws, not diminishing returns.** The improvement per dollar spent does decrease, but it decreases as a power law --- much more slowly than exponential decay. There is no obvious "plateau" in the scaling curves. Each doubling of compute yields a roughly fixed reduction in loss.

**Model size matters more than data (Kaplan's view).** The original Kaplan scaling laws suggested that for a fixed compute budget, it is more efficient to train a larger model on less data than a smaller model on more data. Specifically, Kaplan et al. found that $N$ should scale as $C^{0.73}$ and $D$ should scale as $C^{0.27}$. This led to the GPT-3 regime: very large models trained for relatively few tokens.

**Architecture matters less than scale.** Within the transformer family, the scaling exponents are remarkably consistent across architectural variations (depth vs. width, attention heads, etc.). The absolute loss values differ, but the slopes are similar. This suggests that the scaling behavior is a property of the task (language modeling) rather than the specific architecture.

**Training efficiency.** The relationship between compute and loss implies an irreducible loss $L_{\infty}$ (the entropy of natural language) plus a power-law component. As compute approaches infinity, the loss approaches this irreducible entropy, which reflects genuine uncertainty in language (multiple valid next tokens).

### 11.1.4 Implications for Resource Allocation

Given a fixed compute budget $C$, the scaling laws dictate how to allocate resources between model size and training tokens. If we define $C \approx 6ND$ (approximate FLOPs for a transformer forward + backward pass), then optimizing $L(N, D)$ subject to $6ND = C$ yields the compute-optimal allocation. Kaplan et al. found this to be heavily tilted toward model size, suggesting models should be "large and undertrained" relative to data.

This recommendation was influential --- it directly informed the design of GPT-3 (175B parameters, 300B tokens). However, as we shall see in the next section, this conclusion was significantly revised by subsequent work.

---

## 11.2 Chinchilla Scaling

Hoffmann et al. (2022) at DeepMind conducted a more thorough investigation of compute-optimal scaling, arriving at conclusions that substantially revised Kaplan et al.'s recommendations.

### 11.2.1 The Chinchilla Result

Hoffmann et al. trained over 400 language models ranging from 70M to 16B parameters, varying both model size and the number of training tokens. Their key finding:

**For compute-optimal training, model size and training tokens should scale equally.**

Specifically, for a compute budget $C$:

$$N_{\text{opt}} \propto C^{0.50}, \quad D_{\text{opt}} \propto C^{0.50}$$

This implies approximately **20 tokens per parameter** is optimal. The relationship is:

$$D_{\text{opt}} \approx 20 \cdot N_{\text{opt}}$$

### 11.2.2 Why Chinchilla Differed from Kaplan

The discrepancy arose from methodological differences:

1. **Learning rate schedule**: Kaplan et al. used a fixed cosine schedule for all runs. Hoffmann et al. tuned the learning rate schedule for each run, allowing shorter runs to use higher learning rates. This eliminated a confound that made smaller models appear relatively more efficient.

2. **Last-token evaluation**: Kaplan et al. evaluated loss on the last token of each sequence, while Hoffmann et al. used the average loss across all tokens. This affected the estimated scaling exponents.

3. **Parametric fitting**: Hoffmann et al. used three independent approaches to estimate compute-optimal scaling (fitting loss as a function of $N$ and $D$, fitting IsoFLOP profiles, and fitting parametric loss models) and found consistent results.

### 11.2.3 How Chinchilla Changed the Field

The practical implications were enormous:

**GPT-3 was undertrained.** With 175B parameters and 300B tokens, GPT-3 had a ratio of only 1.7 tokens per parameter --- far below the optimal 20. Under Chinchilla scaling, the same compute budget would have yielded a better model with ~70B parameters trained on ~1.4T tokens.

**Chinchilla proved this empirically.** The 70B-parameter Chinchilla model, trained on 1.4T tokens, outperformed the 280B-parameter Gopher model (trained on 300B tokens) despite using the same compute budget. Smaller but better-trained.

**The era of efficient training.** Post-Chinchilla, models like LLaMA (7B--65B trained on 1--1.4T tokens) and Mistral adopted the principle of training smaller models on more data. This also made inference cheaper (smaller models are faster to serve).

| Model | Parameters | Tokens | Tokens/Param | Compute-Optimal? |
|---|---|---|---|---|
| GPT-3 | 175B | 300B | 1.7 | Undertrained |
| Gopher | 280B | 300B | 1.1 | Severely undertrained |
| Chinchilla | 70B | 1.4T | 20 | Optimal |
| LLaMA-65B | 65B | 1.4T | 21.5 | Approximately optimal |
| LLaMA-7B | 7B | 1T | 143 | Overtrained (intentionally) |

### 11.2.4 The Chinchilla Scaling Law (Formal)

The parametric loss model proposed by Hoffmann et al. is:

$$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

where:
- $E \approx 1.69$ is the irreducible loss (entropy of natural language)
- $A \approx 406.4$, $\alpha \approx 0.34$
- $B \approx 410.7$, $\beta \approx 0.28$

Minimizing $L(N, D)$ subject to the compute constraint $C = 6ND$ yields the optimal allocation. The first-order optimality condition gives:

$$\frac{\partial L}{\partial N} \bigg/ \frac{\partial C}{\partial N} = \frac{\partial L}{\partial D} \bigg/ \frac{\partial C}{\partial D}$$

Solving this yields $N_{\text{opt}}$ and $D_{\text{opt}}$ as functions of $C$.

---

## 11.3 Emergent Capabilities

As language models scale, they exhibit qualitative changes in capability that appear to arise suddenly rather than gradually. Wei et al. (2022) catalogued these "emergent abilities" and sparked a significant debate.

### 11.3.1 What Are Emergent Abilities?

An emergent ability is defined as one that is "not present in smaller models but is present in larger models" (Wei et al., 2022). More precisely, performance on a specific task is near-random for models below a certain scale and then rapidly improves above that threshold.

Examples of claimed emergent abilities:

**Few-shot arithmetic.** Models below ~10B parameters perform near-chance on 3-digit addition, while models above ~100B perform well above chance. The transition is sharp.

**Chain-of-thought reasoning.** Prompting a model to "think step by step" dramatically improves reasoning performance, but this technique is only effective above a certain model scale (~100B parameters for the original findings).

**Word unscrambling.** Models below ~60B parameters cannot unscramble words (e.g., "TENGA" -> "AGENT"), while larger models succeed at well above chance rates.

**Multi-step logical reasoning.** Tasks requiring several logical steps show a sharp phase transition: models either cannot do them at all or can do them reliably.

### 11.3.2 The Controversy: Are Emergences Real?

Schaeffer et al. (2023) challenged the emergence narrative with a provocative claim: **emergent abilities are a mirage created by the choice of evaluation metric**.

Their argument:

1. **Nonlinear metrics create apparent discontinuities.** Many emergence claims use accuracy (an all-or-nothing metric) or exact-match scoring. A model that gradually improves at a task --- say, getting more characters of a multi-digit answer correct --- will show zero accuracy until it gets the entire answer right, then suddenly jump to high accuracy. The underlying capability improves smoothly; the metric creates the appearance of a phase transition.

2. **Continuous metrics show smooth scaling.** When the same tasks are evaluated with continuous metrics (like token-level accuracy, Brier score, or log-likelihood), the improvement is smooth and predictable, with no apparent discontinuity.

3. **Empirical demonstration.** Schaeffer et al. showed that for many "emergent" tasks, switching from exact-match accuracy to a continuous metric eliminates the apparent emergence. Performance improves gradually with scale, exactly as the scaling laws predict.

### 11.3.3 The Resolution (or Lack Thereof)

The truth likely lies in the middle:

**Some "emergences" are metric artifacts.** Tasks where the model must produce an exact multi-token answer (arithmetic, word unscrambling) clearly show smoother improvement with better metrics. The underlying capability scales predictably.

**Some capabilities may genuinely emerge.** In-context learning (the ability to perform novel tasks from few demonstrations), compositional generalization, and certain forms of reasoning appear to involve qualitative changes in how the model processes information. These may represent genuine phase transitions in the model's computational capabilities, analogous to phase transitions in physics.

**The practical distinction matters less than it seems.** Whether a capability "emerges" abruptly or improves smoothly, the practical consequence is the same: there are model sizes below which certain tasks cannot be performed reliably, and model sizes above which they can. For practitioners, the key question is: "How large must my model be to perform this task?" --- and the answer comes from empirical scaling curves, regardless of whether they show sharp or smooth transitions.

### 11.3.4 In-Context Learning: A Case Study

In-context learning (ICL) --- the ability to learn from examples provided in the prompt without any gradient updates --- is perhaps the most remarkable capability of large language models (Brown et al., 2020).

**Zero-shot**: The model performs a task from instructions alone.
**Few-shot**: The model is given several input-output examples and must generalize to a new input.

ICL appears to involve a fundamentally different computational mechanism than training-time learning. Research by Olsson et al. (2022) identified "induction heads" --- specific attention patterns that implement a form of in-context pattern matching --- that emerge during training and are associated with the onset of ICL capability.

Whether ICL counts as "emergent" depends on definition. The underlying mechanism (induction heads) develops gradually during training, but its behavioral manifestation (sudden ability to perform novel tasks from examples) appears more discontinuous.

---

## 11.4 Inference-Time Compute

A paradigm shift has been underway in the LLM field: from scaling training compute to scaling inference compute. Instead of training ever-larger models, recent work explores using more computation at inference time to improve performance.

### 11.4.1 Chain-of-Thought as Test-Time Compute

Chain-of-thought (CoT) prompting (Wei et al., 2022) is the simplest form of inference-time compute scaling. By generating intermediate reasoning steps before the final answer, the model effectively uses additional computation (more generated tokens) to solve the problem.

```
Prompt: What is 23 × 47?

Without CoT: 1081

With CoT: Let me work through this step by step.
23 × 47 = 23 × (50 - 3) = 23 × 50 - 23 × 3 = 1150 - 69 = 1081
```

The CoT tokens serve as a form of working memory, allowing the model to decompose complex problems into manageable steps. Each step involves a simpler computation that the model can perform reliably.

### 11.4.2 Self-Consistency

Self-consistency (Wang et al., 2023) generates multiple chain-of-thought reasoning paths and takes a majority vote on the final answer:

1. Generate $N$ independent CoT solutions (using temperature sampling).
2. Extract the final answer from each solution.
3. Return the most common final answer.

This exploits the observation that different reasoning paths may make different errors, but the correct answer is more likely to be reached by multiple paths. Self-consistency with $N = 40$ typically improves accuracy by 5--15% on reasoning benchmarks compared to single CoT.

The compute cost scales linearly with $N$, and the accuracy improvement follows a logarithmic curve --- similar to best-of-N in RLHF (Section 9.7), with diminishing returns at large $N$.

### 11.4.3 Tree-of-Thought

Tree-of-Thought (Yao et al., 2023) generalizes CoT from a single chain to a tree of reasoning possibilities:

1. At each step, generate multiple candidate next-steps.
2. Evaluate each candidate (using the model itself or a verifier).
3. Expand the most promising candidates.
4. Use search algorithms (BFS, DFS) to explore the tree.

This is especially effective for problems that require search or backtracking --- puzzles, planning, and creative tasks where the first approach may be a dead end.

### 11.4.4 Process Reward Models

Outcome reward models (as used in RLHF) assign a single score to the complete response. Process reward models (PRMs) (Lightman et al., 2023) assign scores to each intermediate step:

$$R_{\text{process}}(x, s_1, s_2, \ldots, s_T) = \prod_{t=1}^{T} P(\text{step } s_t \text{ is correct} | x, s_1, \ldots, s_{t-1})$$

PRMs enable:
- **Step-level verification**: Identify and correct errors in intermediate reasoning.
- **Search guidance**: Use step-level scores to guide tree search (expand steps more likely to be correct).
- **Training signal**: Provide dense feedback for RL training, rather than sparse outcome-level feedback.

Lightman et al. (2023) showed that process supervision significantly outperforms outcome supervision for mathematical reasoning, as it provides a finer-grained learning signal.

### 11.4.5 The Shift from Training to Inference Compute

The relationship between training and inference compute is becoming a core strategic question. Consider two approaches to improving model performance:

**Approach A**: Double the training compute (larger model or more data).
**Approach B**: Keep the same model but use 10x more inference compute (self-consistency, tree search).

For many tasks, Approach B is increasingly favored because:
- Training compute is paid once; inference compute is paid per query.
- Inference compute can be adjusted per query based on difficulty.
- Small models with inference-time search can match large models on specific tasks.

The OpenAI o1/o3 and DeepSeek-R1 model families represent this paradigm: they are trained to use chain-of-thought reasoning extensively, converting inference compute into performance. This shift suggests that the effective compute of a model is not just its training FLOPs but the product of training and inference compute.

---

## 11.5 The Bitter Lesson

Richard Sutton's 2019 essay "The Bitter Lesson" articulates a meta-lesson from 70 years of AI research:

> *"The biggest lesson that can be read from 70 years of AI research is that general methods that leverage computation are ultimately the most effective, and by a large margin."*

### 11.5.1 The Argument

Sutton observes a recurring pattern across AI subfields:

1. **Chess**: Hand-crafted evaluation functions and opening books were eventually overwhelmed by brute-force search (Deep Blue) and then by self-play with minimal domain knowledge (AlphaZero).
2. **Computer vision**: Hand-crafted features (SIFT, HOG) were replaced by learned features (CNNs) trained on large datasets.
3. **Speech recognition**: Hidden Markov Models with hand-crafted features were replaced by end-to-end deep learning.
4. **NLP**: Elaborate linguistic pipelines (POS tagging, parsing, feature engineering) were replaced by pretrained transformers.

The pattern: researchers invest enormous effort in building domain-specific knowledge into systems. Then someone shows that a simple, general method --- given enough compute and data --- outperforms the hand-crafted approach.

### 11.5.2 Implications for LLMs

The Bitter Lesson has profound implications for LLM research:

**Architecture search is less important than scaling.** Elaborate architectural innovations (mixture-of-experts, novel attention patterns, specialized layers) typically provide modest improvements compared to simply making the model larger or training it on more data.

**Data engineering matters more than algorithmic cleverness.** Investing in better data (more diverse, higher quality, better filtered) yields more reliable improvements than novel training algorithms.

**General capabilities beat task-specific optimization.** Rather than building separate models for each task, a single large model trained on diverse data and aligned with human preferences outperforms ensembles of specialized models.

### 11.5.3 Counterpoints and Nuance

The Bitter Lesson is intentionally provocative and has legitimate counterpoints:

- **Efficiency matters.** Compute is finite and expensive. Methods that achieve the same performance with less compute (e.g., Flash Attention, quantization, better architectures) are valuable.
- **Inductive biases accelerate learning.** The transformer architecture itself embodies useful inductive biases (attention, positional encoding). These are not "brute force" --- they are principled design choices that enable efficient scaling.
- **Not all problems scale the same way.** Some tasks may require fundamentally different approaches, not just more compute.

The most productive interpretation: **do not fight scaling, but do invest in making scaling efficient.**

---

## 11.6 Practical Implications

### 11.6.1 Compute-Optimal Model Selection

Given a compute budget $C$ (in FLOPs), the Chinchilla scaling law provides a recipe:

1. Compute the optimal model size: $N_{\text{opt}} \approx 0.0736 \cdot C^{0.50}$ (from Hoffmann et al.)
2. Compute the optimal token count: $D_{\text{opt}} \approx C / (6 \cdot N_{\text{opt}})$
3. Select the nearest practical model size (architectural constraints require certain dimensions).

**Example**: With a budget of $10^{22}$ FLOPs (roughly 1000 A100-hours):
- $N_{\text{opt}} \approx 730\text{M}$ parameters
- $D_{\text{opt}} \approx 15\text{B}$ tokens

### 11.6.2 Estimating Training Costs

The cost of training can be estimated as:

$$\text{Cost} = \frac{C}{\text{GPU throughput} \times \text{GPU utilization} \times \text{hours per dollar}}$$

For an A100 GPU:
- Peak throughput: ~312 TFLOPS (BF16)
- Typical utilization (MFU): 30--50%
- Effective throughput: ~100--156 TFLOPS
- Cost: ~$2/hour (cloud pricing)

**Example**: Training a 7B model on 1T tokens:
- $C \approx 6 \times 7 \times 10^9 \times 10^{12} = 4.2 \times 10^{22}$ FLOPs
- At 130 TFLOPS effective: $4.2 \times 10^{22} / (130 \times 10^{12}) = 3.2 \times 10^8$ seconds
- GPU-hours: $\approx 90,000$
- With 1000 GPUs: ~90 hours (3.75 days)
- Cost at $2/GPU-hour: ~$180,000

These are rough estimates. Actual costs include communication overhead, checkpoint I/O, data loading, and infra costs.

### 11.6.3 When to Use Bigger Models vs. More Data

The answer depends on the deployment scenario:

**Training-compute-limited** (one-time training budget): Use Chinchilla scaling --- balance model size and data.

**Inference-compute-limited** (serving at scale): Overtrain a smaller model. LLaMA-7B was trained on 1T tokens (143 tokens per parameter, 7x the Chinchilla optimum). This "wastes" training compute but produces a smaller, cheaper-to-serve model.

**Data-limited** (domain-specific corpus is small): Use a larger model with less data. The model's capacity allows it to learn efficiently from fewer examples.

**Quality-limited** (data quality is the bottleneck): Invest in data quality rather than quantity. Better filtering, deduplication, and curation yield larger improvements than more raw data.

---

## 11.7 Beyond Chinchilla

### 11.7.1 Inference-Adjusted Scaling

Sardana & Frankle (2023) proposed inference-adjusted scaling laws that account for deployment costs. The key insight: a Chinchilla-optimal model minimizes training compute for a given loss level, but the total cost of a model includes both training and inference. If the model will be queried millions of times, a smaller model (cheaper per query) trained for longer (more expensive to train) may minimize total cost.

The inference-adjusted compute-optimal model size is:

$$N_{\text{inference-opt}} \propto \frac{N_{\text{Chinchilla}}}{(1 + R)^{1/(1+\alpha/\beta)}}$$

where $R$ is the ratio of total inference FLOPs to training FLOPs. For widely deployed models ($R \gg 1$), the optimal model is significantly smaller than Chinchilla optimal.

### 11.7.2 LLaMA and Efficiency-Focused Scaling

The LLaMA models (Touvron et al., 2023) explicitly adopted an efficiency-focused scaling philosophy: train smaller models for longer than Chinchilla recommends, optimizing for inference cost rather than training cost.

- LLaMA-7B: 1T tokens (143 tokens/param) --- 7x overtraining
- LLaMA-13B: 1T tokens (77 tokens/param) --- 3.8x overtraining
- LLaMA-65B: 1.4T tokens (21.5 tokens/param) --- approximately Chinchilla-optimal

This strategy proved immensely practical: the 7B and 13B models could run on consumer hardware while delivering strong performance, catalyzing the open-source LLM ecosystem.

### 11.7.3 Overtrained Small Models

The trend toward overtraining small models has accelerated:

| Model | Parameters | Tokens | Tokens/Param |
|---|---|---|---|
| Chinchilla | 70B | 1.4T | 20 |
| LLaMA-7B | 7B | 1T | 143 |
| Mistral-7B | 7B | ~8T | ~1143 |
| Phi-2 | 2.7B | 1.4T | 519 |
| LLaMA 3 8B | 8B | 15T | 1875 |

These models demonstrate that Chinchilla optimality is a lower bound on useful training, not an upper bound. When inference cost matters (which it almost always does in production), overtraining is the rational choice.

### 11.7.4 Data Quality as a Scaling Axis

Recent work (Gunasekar et al., 2023; Li et al., 2023) has shown that data quality functions as an independent scaling axis. The Phi series of models demonstrated that small models trained on "textbook-quality" synthetic data can match much larger models trained on web crawl. This suggests a modified scaling law:

$$L(N, D, Q) = E + \frac{A}{N^\alpha} + \frac{B}{(D \cdot Q)^\beta}$$

where $Q$ is a data quality factor. High-quality data is equivalent to having more data --- a 10x improvement in quality might be worth 10x more tokens.

This is an active area of research, and the exact quantification of "quality" remains elusive. But the practical lesson is clear: investing in data quality can be more cost-effective than scaling compute.

---

## 11.8 Summary

Scaling laws provide a quantitative framework for understanding and predicting language model performance. The key takeaways:

1. **Performance scales as a power law** with model size, data, and compute. There are no obvious plateaus.
2. **Chinchilla scaling** established that model size and data should scale equally for compute-optimal training (~20 tokens per parameter).
3. **In practice, overtraining small models** is often preferable because inference costs dominate total costs for deployed models.
4. **Emergent capabilities** may or may not represent genuine phase transitions, but the practical consequence is the same: some tasks require a minimum model scale.
5. **Inference-time compute** is an increasingly important dimension of scaling, complementing training-time compute.
6. **The Bitter Lesson** tells us to invest in methods that leverage computation, not in methods that cleverly avoid it.
7. **Data quality** functions as a multiplier on effective data quantity, offering a potentially more efficient scaling axis than raw compute.

These principles are not merely academic. They guide decisions worth millions of dollars: how large a model to train, how much data to collect, when to stop training, and how to allocate inference budgets. A practitioner who internalizes these scaling laws can make informed decisions about model selection, training planning, and resource allocation --- avoiding both underspending (training a model too small to perform the target task) and overspending (training a model far larger than necessary).

---

## Exercises

1. **Scaling Law Estimation**: Train a family of language models at 5 different scales (e.g., 1M, 10M, 50M, 100M, 300M parameters) on the same dataset. Plot validation loss vs. model size on a log-log scale. Does a power-law relationship hold? Fit the scaling exponent and compare to Kaplan et al.'s reported value.

2. **Chinchilla Prediction**: Using the Chinchilla scaling law ($D_{\text{opt}} \approx 20N$), calculate the compute-optimal configuration for training budgets of $10^{19}$, $10^{20}$, $10^{21}$, $10^{22}$, and $10^{23}$ FLOPs. For each, specify the model size, token count, and estimated training time on a cluster of 256 A100 GPUs.

3. **Emergence Investigation**: Select a task claimed to exhibit emergence (e.g., multi-digit arithmetic). Evaluate models of varying sizes using both exact-match accuracy and a continuous metric (token-level accuracy or log-likelihood). Does the apparent emergence disappear with the continuous metric?

4. **Inference-Time Compute**: Implement self-consistency for a mathematical reasoning task (e.g., GSM8K). Plot accuracy vs. number of samples $N$ for $N \in \{1, 5, 10, 20, 40, 100\}$. At what point do diminishing returns set in?

5. **Cost Analysis**: You need to deploy an LLM that achieves perplexity < 10 on a domain-specific corpus. Using published scaling laws, estimate: (a) the minimum model size, (b) the training data requirement, (c) the training cost on cloud GPUs, and (d) the inference cost per 1000 queries. Compare a Chinchilla-optimal configuration vs. an overtrained small model configuration.

6. **Bitter Lesson Debate**: Write a structured argument (2--3 pages) either supporting or critiquing the Bitter Lesson in the context of LLMs. Consider: Is the transformer architecture a form of "general method"? Does RLHF/DPO alignment represent human-knowledge injection? Does Flash Attention violate the spirit of the Bitter Lesson?

---

## References

- Brown, T. B., et al. (2020). Language Models are Few-Shot Learners. *Advances in Neural Information Processing Systems, 33*, 1877--1901.
- Gunasekar, S., et al. (2023). Textbooks Are All You Need. *arXiv preprint arXiv:2306.11644*.
- Hoffmann, J., et al. (2022). Training Compute-Optimal Large Language Models. *Advances in Neural Information Processing Systems, 35*.
- Kaplan, J., et al. (2020). Scaling Laws for Neural Language Models. *arXiv preprint arXiv:2001.08361*.
- Li, Y., et al. (2023). Textbooks Are All You Need II: phi-1.5 Technical Report. *arXiv preprint arXiv:2309.05463*.
- Lightman, H., et al. (2023). Let's Verify Step by Step. *arXiv preprint arXiv:2305.20050*.
- Olsson, C., et al. (2022). In-context Learning and Induction Heads. *Transformer Circuits Thread*.
- Sardana, N., & Frankle, J. (2023). Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws. *arXiv preprint arXiv:2401.00448*.
- Schaeffer, R., Miranda, B., & Koyejo, S. (2023). Are Emergent Abilities of Large Language Models a Mirage? *Advances in Neural Information Processing Systems, 36*.
- Sutton, R. (2019). The Bitter Lesson. *Personal blog post*. http://www.incompleteideas.net/IncIdeas/BitterLesson.html
- Touvron, H., et al. (2023). LLaMA: Open and Efficient Foundation Language Models. *arXiv preprint arXiv:2302.13971*.
- Wang, X., et al. (2023). Self-Consistency Improves Chain of Thought Reasoning in Language Models. *Proceedings of ICLR 2023*.
- Wei, J., et al. (2022). Emergent Abilities of Large Language Models. *Transactions on Machine Learning Research*.
- Yao, S., et al. (2023). Tree of Thoughts: Deliberate Problem Solving with Large Language Models. *Advances in Neural Information Processing Systems, 36*.
