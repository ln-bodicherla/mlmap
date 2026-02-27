# Chapter 9: Alignment --- From RLHF to DPO

> *"The question of whether a computer can think is no more interesting than the question of whether a submarine can swim."* -- Edsger Dijkstra

## Learning Objectives

By the end of this chapter, you will be able to:

1. Articulate why alignment is necessary and what the gap between pretraining and useful behavior entails.
2. Design instruction-tuning datasets and understand the impact of data format and quality on model behavior.
3. Derive the Bradley-Terry preference model and train a reward model from human preference data.
4. Explain the full RLHF pipeline --- SFT, reward modeling, and PPO optimization --- including the mathematical details of the clipped surrogate objective and KL penalty.
5. Derive the DPO loss function from first principles, starting from the RLHF objective.
6. Compare alignment methods (RLHF, DPO, Constitutional AI, rejection sampling) and select the appropriate approach for a given setting.
7. Evaluate aligned models using contemporary benchmarks and understand their limitations.

---

## 9.1 The Alignment Problem

A large language model, fresh from pretraining, is a remarkably capable but fundamentally undirected system. It has learned to predict the next token given a context --- nothing more. When prompted with a question, it may answer correctly, or it may continue the text as if it were a web document, a piece of fiction, or a forum post. It has no inherent preference for being helpful, truthful, or safe.

**Alignment** is the process of steering a model's behavior to match human intentions and values. More precisely, we want the model to be:

- **Helpful**: It should follow instructions and provide useful, relevant responses.
- **Honest**: It should not fabricate information and should express appropriate uncertainty.
- **Harmless**: It should refuse to assist with dangerous requests and avoid generating toxic content.

This triad --- helpful, honest, harmless (HHH) --- was articulated by Askell et al. (2021) and has become the standard framework for thinking about alignment.

### 9.1.1 Why Instruction-Following Matters

The pretraining objective optimizes for next-token prediction: $\max_\theta \sum_t \log P_\theta(x_t | x_{<t})$. This objective rewards the model for modeling the training distribution faithfully. But the training distribution contains everything: well-written technical articles, hateful forum posts, outdated misinformation, creative fiction, and raw code. A model that perfectly models this distribution is not one that a user would want to interact with.

The fundamental insight of the alignment literature is that there is a **distributional mismatch** between "text likely to follow this context on the internet" and "text that a human would find helpful in response to this query." Alignment techniques bridge this gap.

### 9.1.2 A Brief History

The modern alignment pipeline was pioneered by OpenAI with InstructGPT (Ouyang et al., 2022), which demonstrated that a relatively small amount of human feedback could dramatically improve the perceived quality of a language model. The pipeline consisted of three stages:

1. **Supervised Fine-Tuning (SFT)** on human-written demonstrations.
2. **Reward Model (RM) training** on human preference comparisons.
3. **Reinforcement Learning from Human Feedback (RLHF)** using Proximal Policy Optimization (PPO).

This three-stage pipeline transformed GPT-3, a capable but unreliable completion model, into InstructGPT, a model that users consistently preferred. Subsequent work has refined and simplified this pipeline, most notably through Direct Preference Optimization (DPO) (Rafailov et al., 2023), which eliminates the need for a separate reward model and RL training.

---

## 9.2 Instruction Tuning

The first step in alignment is instruction tuning (also called supervised fine-tuning or SFT): training the model on examples of desired input-output behavior.

### 9.2.1 FLAN and the Power of Task Diversity

The Finetuned Language Net (FLAN) work by Chung et al. (2022) demonstrated a powerful principle: **instruction-tuning on a diverse mixture of tasks improves generalization to unseen tasks**. FLAN-T5 and FLAN-PaLM were trained on over 1,800 tasks phrased as natural language instructions, spanning translation, summarization, question answering, classification, reasoning, and more.

The key finding was that task diversity mattered more than task volume. A model trained on many diverse tasks (even with few examples per task) generalized better than one trained on many examples of few tasks. This is because diverse instruction tuning teaches the model the meta-skill of "following instructions" rather than memorizing specific task patterns.

### 9.2.2 Instruction Data Formats

Several formats have become standard in the field:

**Alpaca Format** (Taori et al., 2023):
```
### Instruction:
{instruction text}

### Input:
{optional input text}

### Response:
{desired response}
```

**ShareGPT/Chat Format**:
```json
[
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "What causes the northern lights?"},
  {"role": "assistant", "content": "The northern lights, or aurora borealis..."}
]
```

**ChatML Format** (used by many modern models):
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What causes the northern lights?<|im_end|>
<|im_start|>assistant
The northern lights, or aurora borealis...<|im_end|>
```

The choice of format matters because the model must learn the conversational structure --- when to respond, when to stop, and how to handle multi-turn context. Using special tokens (like `<|im_start|>` and `<|im_end|>`) provides unambiguous boundaries, which is particularly important during inference.

### 9.2.3 Data Quality Over Quantity

A recurring theme in alignment research is that **data quality dramatically outweighs data quantity**. The LIMA paper (Zhou et al., 2023) made this point forcefully: fine-tuning LLaMA-65B on just 1,000 carefully curated instruction-response pairs produced a model that rivaled much more extensively trained alternatives.

This "less is more" phenomenon arises because instruction tuning primarily teaches the model **format and style** rather than new knowledge. The model already possesses vast knowledge from pretraining; SFT teaches it to present this knowledge in the format users expect. A small number of high-quality, diverse examples is sufficient to induce this behavioral shift.

Quality indicators for instruction-tuning data include:

- **Response correctness**: Factually accurate and logically sound.
- **Response completeness**: Addresses all aspects of the instruction.
- **Appropriate tone**: Professional, clear, and appropriately detailed.
- **Diversity**: Covers a wide range of topics, difficulties, and response lengths.
- **No contamination**: Test set answers or benchmark solutions must be excluded.

---

## 9.3 Reward Modeling

After SFT, the model can follow instructions, but its responses may still be suboptimal --- verbose, shallow, occasionally wrong, or stylistically inconsistent. Reward modeling provides a scalar signal that quantifies response quality, enabling further optimization.

### 9.3.1 The Bradley-Terry Model

The Bradley-Terry model (Bradley & Terry, 1952) provides the mathematical foundation for learning from pairwise preferences. Given two responses $y_1$ and $y_2$ to a prompt $x$, a human annotator indicates which response they prefer. The Bradley-Terry model assumes:

$$P(y_1 \succ y_2 | x) = \sigma(r(x, y_1) - r(x, y_2))$$

where $r(x, y)$ is the reward (or quality score) of response $y$ to prompt $x$, $\sigma$ is the sigmoid function, and $y_1 \succ y_2$ denotes that $y_1$ is preferred over $y_2$.

**Derivation.** Assume each response has a latent quality $r(x, y)$ and the observed preference is a noisy observation of this quality difference. If the noise follows a logistic distribution (a standard assumption), then the probability of observing the preference $y_1 \succ y_2$ is exactly the sigmoid of the quality difference.

### 9.3.2 Training the Reward Model

The reward model $r_\phi(x, y)$ is typically initialized from the SFT model (with the language modeling head replaced by a scalar output head) and trained to minimize the negative log-likelihood of the observed preferences:

$$\mathcal{L}_{\text{RM}}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l)) \right]$$

where $y_w$ is the preferred (winning) response and $y_l$ is the dispreferred (losing) response.

```python
import torch
import torch.nn as nn

class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.reward_head = nn.Linear(base_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Use last token's hidden state as the sequence representation
        last_hidden = outputs.hidden_states[-1]
        # Get the last non-padding token
        sequence_lengths = attention_mask.sum(dim=1) - 1
        last_token_hidden = last_hidden[
            torch.arange(last_hidden.size(0)), sequence_lengths
        ]
        reward = self.reward_head(last_token_hidden).squeeze(-1)
        return reward

def compute_reward_loss(reward_model, chosen_ids, chosen_mask,
                        rejected_ids, rejected_mask):
    r_chosen = reward_model(chosen_ids, chosen_mask)
    r_rejected = reward_model(rejected_ids, rejected_mask)
    loss = -torch.log(torch.sigmoid(r_chosen - r_rejected)).mean()
    return loss
```

### 9.3.3 Preference Data Collection

Collecting preference data is the most expensive and consequential step. Several approaches exist:

**Human labeling.** Trained annotators compare pairs of model outputs and indicate which is better. This is the gold standard but expensive ($5--$25 per comparison). InstructGPT used approximately 50,000 human comparisons. Key design decisions include:

- **Number of responses per prompt**: Typically 2--4 responses are generated, and annotators rank or compare them pairwise.
- **Annotator agreement**: Inter-annotator agreement is typically 60--75%, reflecting the genuine subjectivity of quality judgments.
- **Annotation guidelines**: Clear, detailed guidelines dramatically improve consistency.

**AI-assisted labeling.** A strong model (e.g., GPT-4) can serve as a preference judge, dramatically reducing cost. This approach is sometimes called RLAIF (RL from AI Feedback) and is a component of Constitutional AI (Section 9.6). While cheaper, it introduces the risk of amplifying the judge model's biases.

**Implicit feedback.** User interactions (upvotes, regeneration requests, conversation length) provide weak but plentiful preference signals. These are noisier than explicit comparisons but can be collected at scale.

### 9.3.4 Evaluating the Reward Model

A reward model is only useful if it accurately reflects human preferences. Key evaluation metrics include:

- **Accuracy**: Fraction of held-out preference pairs where the reward model agrees with the human label. Typical values are 65--75%.
- **Calibration**: Does $P(y_1 \succ y_2) = 0.7$ actually mean $y_1$ is preferred 70% of the time?
- **Robustness**: Is the reward model fooled by verbose but unhelpful responses? This failure mode, called "reward hacking," is a persistent challenge.

---

## 9.4 RLHF: Reinforcement Learning from Human Feedback

With a trained reward model in hand, we can optimize the language model's policy to maximize reward. This is the RL stage of RLHF, and it uses Proximal Policy Optimization (PPO) (Schulman et al., 2017).

### 9.4.1 The RLHF Objective

The goal is to find a policy $\pi_\theta$ that maximizes the expected reward while staying close to the SFT model (the reference policy $\pi_{\text{ref}}$):

$$\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)} \left[ r_\phi(x, y) \right] - \beta \cdot D_{\text{KL}}\left[ \pi_\theta(\cdot|x) \| \pi_{\text{ref}}(\cdot|x) \right]$$

The KL divergence penalty serves two critical purposes:

1. **Prevents reward hacking**: Without the KL constraint, the policy would find degenerate outputs that exploit weaknesses in the reward model (e.g., extremely long, repetitive responses that accidentally score highly).
2. **Preserves capabilities**: The SFT model has useful capabilities (coherent generation, factual knowledge) that we do not want to lose during RL optimization.

The hyperparameter $\beta$ controls the strength of the constraint. Higher $\beta$ keeps the policy closer to the reference, sacrificing some reward for stability. Lower $\beta$ allows more deviation, potentially achieving higher reward but risking reward hacking.

### 9.4.2 PPO Algorithm

PPO (Schulman et al., 2017) is the workhorse of RLHF. It is a policy gradient method that uses a clipped surrogate objective to prevent destructively large policy updates.

**The policy gradient.** The basic policy gradient estimator is:

$$\nabla_\theta J(\theta) = \mathbb{E}_t \left[ \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot \hat{A}_t \right]$$

where $\hat{A}_t$ is the advantage estimate --- how much better action $a_t$ was compared to the expected value at state $s_t$.

**The clipped surrogate objective.** Let $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ be the probability ratio between the new and old policies. PPO optimizes:

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

where $\epsilon$ (typically 0.2) limits how much the probability ratio can change. This clipping prevents the policy from moving too far in a single update, ensuring stable training.

**The value function.** PPO also trains a value function $V_\psi(s)$ that estimates the expected cumulative reward from state $s$. The advantage is estimated using Generalized Advantage Estimation (GAE):

$$\hat{A}_t = \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l}$$

where $\delta_t = r_t + \gamma V_\psi(s_{t+1}) - V_\psi(s_t)$ is the temporal difference error.

### 9.4.3 RLHF in Practice: The Language Model Setting

In the language model setting, the correspondence to RL concepts is:

| RL Concept | Language Model Equivalent |
|---|---|
| State $s_t$ | Prompt + tokens generated so far |
| Action $a_t$ | Next token to generate |
| Policy $\pi_\theta$ | Language model |
| Reward | Reward model score on complete response, minus KL penalty |
| Episode | Generating a complete response to a prompt |

The per-token reward structure is:

$$r_t = \begin{cases}
-\beta \log \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{ref}}(a_t|s_t)} & \text{for } t < T \text{ (intermediate tokens)} \\
r_\phi(x, y) - \beta \log \frac{\pi_\theta(a_T|s_T)}{\pi_{\text{ref}}(a_T|s_T)} & \text{for } t = T \text{ (final token)}
\end{cases}$$

The KL penalty is applied at every token to prevent drift from the reference model throughout the generation process, not just at the end.

### 9.4.4 The InstructGPT Architecture

The InstructGPT paper (Ouyang et al., 2022) provided the canonical instantiation of this pipeline:

1. **SFT Stage**: GPT-3 (175B) fine-tuned on ~13K human-written demonstrations.
2. **RM Stage**: A 6B reward model trained on ~33K human preference comparisons.
3. **PPO Stage**: The SFT model optimized against the reward model using PPO, with a KL penalty coefficient $\beta = 0.02$.

The key result: the 1.3B InstructGPT model was preferred by human labelers over the 175B GPT-3, despite being over 100x smaller. This demonstrated that alignment is not just about scale --- it is about optimization toward human preferences.

### 9.4.5 Challenges of RLHF

Despite its success, RLHF has significant practical challenges:

- **Complexity**: Four models must be loaded simultaneously during PPO training (policy, reference, reward, value function), requiring substantial GPU memory.
- **Instability**: PPO training is notoriously sensitive to hyperparameters. Small changes in $\beta$, $\epsilon$, learning rate, or batch size can cause training collapse.
- **Reward hacking**: The policy may find ways to exploit the reward model that do not correspond to genuinely better responses.
- **Cost**: Human preference data is expensive to collect and requires careful quality control.

---

## 9.5 DPO: Direct Preference Optimization

Direct Preference Optimization (Rafailov et al., 2023) eliminates the need for both a reward model and RL training, dramatically simplifying the alignment pipeline. It achieves this through a beautiful mathematical insight.

### 9.5.1 The Key Insight

Starting from the RLHF objective:

$$\max_{\pi} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi(\cdot|x)} \left[ r(x, y) \right] - \beta \cdot D_{\text{KL}}\left[ \pi(\cdot|x) \| \pi_{\text{ref}}(\cdot|x) \right]$$

Rafailov et al. (2023) observed that this constrained optimization problem has a **closed-form solution** for the optimal policy:

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r(x, y)\right)$$

where $Z(x) = \sum_y \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r(x, y)\right)$ is the partition function.

### 9.5.2 Full Mathematical Derivation

**Step 1: Express the reward in terms of the optimal policy.** Rearranging the closed-form solution:

$$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$

**Step 2: Substitute into the Bradley-Terry preference model.** Recall that:

$$P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))$$

Substituting the expression for $r$:

$$P(y_w \succ y_l | x) = \sigma\left(\beta \log \frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)$$

Note that the partition function $Z(x)$ cancels! This is the crucial simplification.

**Step 3: Form the DPO loss.** We can directly optimize the policy $\pi_\theta$ to fit the preference data without ever computing a reward:

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right) \right]$$

This is a supervised learning loss! It requires only:
1. A dataset of preferences $(x, y_w, y_l)$.
2. The ability to compute log-probabilities under $\pi_\theta$ and $\pi_{\text{ref}}$.

No reward model. No RL. No value function. No PPO hyperparameters.

### 9.5.3 Intuition Behind DPO

The DPO loss has an elegant interpretation. Define the **implicit reward** as:

$$\hat{r}(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$$

This is the log-ratio of the policy probability to the reference probability, scaled by $\beta$. DPO increases this implicit reward for preferred responses and decreases it for dispreferred responses.

The gradient of the DPO loss is:

$$\nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta \mathbb{E} \left[ \underbrace{\sigma(-\hat{r}_w + \hat{r}_l)}_{\text{weighting}} \left[ \nabla_\theta \log \pi_\theta(y_w|x) - \nabla_\theta \log \pi_\theta(y_l|x) \right] \right]$$

The weighting term $\sigma(-\hat{r}_w + \hat{r}_l)$ is large when the model currently assigns higher implicit reward to the losing response than the winning one --- i.e., when the model is most "wrong." This means DPO focuses its gradient updates on the examples where the model most disagrees with human preferences.

### 9.5.4 The Beta Parameter

The temperature parameter $\beta$ controls how far the policy can deviate from the reference:

- **Large $\beta$** (e.g., 0.5--1.0): Strong KL constraint. The policy stays close to the reference. Training is stable but alignment may be weak.
- **Small $\beta$** (e.g., 0.01--0.05): Weak KL constraint. The policy can deviate substantially. Stronger alignment but risk of overfitting to preference data.
- **Typical range**: $\beta = 0.1$--$0.5$ works well in most settings.

### 9.5.5 The Reference Model

The reference model $\pi_{\text{ref}}$ is typically the SFT model. During training, it is kept frozen and used only to compute log-probabilities. Its role is to anchor the policy, preventing it from degenerating into a distribution that merely maximizes the preference signal.

### 9.5.6 Implementation

```python
import torch
import torch.nn.functional as F

def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             reference_chosen_logps, reference_rejected_logps,
             beta=0.1):
    """
    Compute the DPO loss.

    Args:
        policy_chosen_logps: Log P(y_w | x) under the policy model
        policy_rejected_logps: Log P(y_l | x) under the policy model
        reference_chosen_logps: Log P(y_w | x) under the reference model
        reference_rejected_logps: Log P(y_l | x) under the reference model
        beta: Temperature parameter

    Returns:
        loss: Scalar DPO loss
        chosen_rewards: Implicit rewards for chosen responses
        rejected_rewards: Implicit rewards for rejected responses
    """
    # Compute implicit rewards
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)

    # DPO loss
    logits = chosen_rewards - rejected_rewards
    loss = -F.logsigmoid(logits).mean()

    return loss, chosen_rewards.detach(), rejected_rewards.detach()
```

Using TRL:

```python
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("sft_model_path")
ref_model = AutoModelForCausalLM.from_pretrained("sft_model_path")
tokenizer = AutoTokenizer.from_pretrained("sft_model_path")

dpo_config = DPOConfig(
    output_dir="./dpo_output",
    beta=0.1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-7,            # Very low LR for DPO
    num_train_epochs=1,            # Usually 1 epoch is sufficient
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=10,
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=preference_dataset,   # Must have "chosen" and "rejected" columns
    tokenizer=tokenizer,
)

trainer.train()
```

### 9.5.7 DPO vs. RLHF: Practical Comparison

| Aspect | RLHF (PPO) | DPO |
|---|---|---|
| Models needed | 4 (policy, reference, reward, value) | 2 (policy, reference) |
| Training algorithm | RL (online sampling + optimization) | Supervised learning |
| Stability | Sensitive to hyperparameters | Much more stable |
| Memory | Very high (4 models) | High (2 models) |
| Data efficiency | Can generate new data online | Fixed offline dataset |
| Performance ceiling | Potentially higher (online exploration) | Slightly lower in some settings |
| Implementation complexity | Very high | Low |
| Iteration speed | Slow (generation + optimization) | Fast (standard training loop) |

In practice, DPO has become the preferred method for most open-source alignment efforts due to its simplicity and stability. RLHF retains advantages in settings where online data generation and iterative preference collection are feasible (typically well-resourced industry labs).

---

## 9.6 Constitutional AI

Constitutional AI (CAI), introduced by Bai et al. (2022), takes a different approach to alignment. Instead of relying on human preference labels, CAI uses a set of **principles** (a "constitution") and trains the model to critique and revise its own responses according to these principles.

### 9.6.1 The Constitutional Approach

CAI proceeds in two phases:

**Phase 1: Supervised Learning from AI Feedback (Critique-Revision)**

1. Generate responses to prompts, including potentially harmful ones.
2. Ask the model to **critique** its own response according to a specific constitutional principle (e.g., "Is this response harmful? Could it be improved to be more helpful while being harmless?").
3. Ask the model to **revise** its response based on the critique.
4. Fine-tune the model on the revised responses.

Example principle: "Please choose the response that is the most helpful, honest, and harmless."

**Phase 2: RLAIF (RL from AI Feedback)**

1. Generate pairs of responses to prompts.
2. Use the constitutional model (from Phase 1) to select the preferred response.
3. Train a reward model on these AI-generated preferences.
4. Optimize the policy using RL (PPO) against this reward model.

### 9.6.2 Advantages and Limitations

**Advantages:**
- Dramatically reduces the need for human labeling.
- Principles are explicit and auditable --- one can inspect and modify the constitution.
- Scales better than human feedback: the AI judge can evaluate millions of response pairs.

**Limitations:**
- The quality of alignment is bounded by the model's ability to evaluate its own responses.
- Principles may conflict in ambiguous situations.
- May amplify biases present in the base model's judgment.

CAI represents Anthropic's approach to alignment and underpins the Claude model family. It demonstrates that transparency about alignment goals (through an explicit constitution) can be both practical and beneficial.

---

## 9.7 Rejection Sampling

Rejection sampling is a simpler alternative to RL that uses the reward model as a filter rather than as a training signal.

### 9.7.1 Best-of-N Sampling

The simplest form of rejection sampling is Best-of-N (BoN):

1. For each prompt $x$, generate $N$ candidate responses $\{y_1, \ldots, y_N\}$ from the policy.
2. Score each response using the reward model: $r_\phi(x, y_i)$.
3. Select the response with the highest reward: $y^* = \arg\max_i r_\phi(x, y_i)$.

Best-of-N is remarkably effective. With $N = 64$, the selected response typically exceeds the quality achievable by PPO, albeit at $64\times$ the inference cost.

The expected improvement follows from order statistics. If reward scores follow a distribution with CDF $F$, the expected maximum of $N$ samples is:

$$\mathbb{E}[\max(r_1, \ldots, r_N)] = \int_0^1 F^{-1}(u^{1/N}) \, du$$

For normally distributed rewards, this grows approximately as $\sqrt{2 \ln N}$, yielding diminishing returns as $N$ increases.

### 9.7.2 Iterative Rejection Sampling Fine-Tuning

A more sophisticated approach uses rejection sampling to generate training data iteratively:

1. Sample $N$ responses per prompt from the current policy.
2. Select the best responses (top-$k$ or above a threshold) using the reward model.
3. Fine-tune the policy on the selected responses.
4. Repeat from step 1.

This approach, used in LLaMA 2's alignment (Touvron et al., 2023), avoids the instabilities of PPO while capturing some of the benefits of online RL (the training distribution evolves as the policy improves). Each iteration the policy generates better candidates, and the best of those become the next training data.

```python
def rejection_sampling_iteration(model, reward_model, prompts,
                                  n_samples=16, top_k=4):
    """One iteration of rejection sampling fine-tuning."""
    training_data = []

    for prompt in prompts:
        # Generate N candidate responses
        candidates = []
        for _ in range(n_samples):
            response = model.generate(prompt, do_sample=True, temperature=0.8)
            reward = reward_model.score(prompt, response)
            candidates.append((response, reward))

        # Select top-k by reward
        candidates.sort(key=lambda x: x[1], reverse=True)
        for response, _ in candidates[:top_k]:
            training_data.append({"prompt": prompt, "response": response})

    return training_data
```

---

## 9.8 Building Instruction Datasets

The quality of alignment is fundamentally bounded by the quality of the training data. This section discusses practical strategies for building instruction-tuning and preference datasets.

### 9.8.1 Data Collection Strategies

**Human-written demonstrations.** Expert annotators write high-quality responses to a diverse set of prompts. This is the highest quality but most expensive approach. Key considerations:

- Recruit domain experts for specialized topics (coding, math, medicine).
- Provide detailed guidelines with examples of desired style and depth.
- Include multi-turn conversations, not just single-turn Q&A.

**Self-instruct and synthetic data.** Use a strong model to generate instruction-response pairs (Wang et al., 2023). The Alpaca dataset was generated by prompting GPT-3.5 with 175 seed tasks, generating 52K instruction-response pairs for under $500. While cost-effective, synthetic data carries risks:

- Quality ceiling: the generated data can not exceed the teacher model's capabilities.
- Diversity limitations: synthetic data often lacks the creativity and edge cases found in human-written data.
- Potential for systematic errors or hallucinations.

**Distillation from strong models.** Generate training data by querying a powerful model (e.g., GPT-4) with diverse prompts. This is a form of knowledge distillation and is used extensively in the open-source community. Legal and ethical considerations apply, as model providers' terms of service may restrict using outputs for training competing models.

**Community contributions.** Platforms like OpenAssistant and Dolly collected volunteer-generated instruction data. These datasets offer authentic human writing but require careful quality control and deduplication.

### 9.8.2 Quality Filtering

Raw data requires extensive filtering:

1. **Deduplication**: Remove exact and near-duplicate instruction-response pairs.
2. **Length filtering**: Remove extremely short responses (likely incomplete) and extremely long ones (likely rambling or off-topic).
3. **Language detection**: Ensure responses are in the target language.
4. **Toxicity filtering**: Use classifiers to remove toxic, biased, or unsafe content.
5. **Factual verification**: For factual questions, verify a sample of responses against trusted sources.
6. **Format consistency**: Ensure all examples follow the chosen template format.

### 9.8.3 Decontamination

Benchmark contamination --- the presence of test examples in training data --- is a pervasive problem. Decontamination involves:

1. Collecting all text from evaluation benchmarks (MMLU, HumanEval, GSM8K, etc.).
2. Computing n-gram overlap between training examples and benchmark instances.
3. Removing training examples with high overlap (typically 13-gram or higher).
4. Documenting the decontamination process for reproducibility.

### 9.8.4 Preference Data Construction

For DPO and RLHF, we need paired preference data. Construction approaches include:

- **Model-generated pairs**: Generate two responses to each prompt using different sampling parameters or different model checkpoints, then have annotators compare them.
- **Rating-derived pairs**: Annotators rate individual responses on a Likert scale. Pairs are formed from responses with different ratings.
- **AI-assisted ranking**: Use a strong model to rank multiple responses, then form pairs from the ranking. This can be augmented with human verification on a subset.

The UltraFeedback dataset (Cui et al., 2023) provides an example of large-scale preference data construction: 64K prompts, each with 4 model responses scored by GPT-4 on helpfulness, honesty, instruction-following, and truthfulness.

---

## 9.9 Evaluation of Aligned Models

Evaluating alignment is inherently challenging because "good" responses are subjective. The field has developed several complementary evaluation approaches.

### 9.9.1 MT-Bench

MT-Bench (Zheng et al., 2023) evaluates multi-turn conversation ability across 8 categories: writing, roleplay, reasoning, math, coding, extraction, STEM, and humanities. It uses GPT-4 as a judge to rate responses on a scale of 1--10.

**Strengths**: Tests multi-turn coherence, covers diverse abilities, reproducible.
**Limitations**: Dependent on GPT-4's judgment, may favor verbose or stylistically similar responses.

### 9.9.2 AlpacaEval

AlpacaEval (Li et al., 2023) compares model responses against a reference model (GPT-4) on 805 instructions. An LLM judge determines which response is preferred. AlpacaEval 2.0 uses a length-controlled metric to avoid rewarding verbosity.

**Strengths**: Fast, reproducible, strong correlation with human preferences.
**Limitations**: Single-turn only, biased toward the reference model's style.

### 9.9.3 Chatbot Arena

Chatbot Arena (Zheng et al., 2023) is a crowdsourced platform where users interact with two anonymous models simultaneously and vote for the better response. Models are ranked using an Elo rating system.

**Strengths**: Real user interactions, diverse prompts, no systematic biases from LLM judges.
**Limitations**: Popularity bias (more popular models get more votes), non-reproducible (different users ask different questions), influenced by model availability.

### 9.9.4 MMLU (Massive Multitask Language Understanding)

MMLU (Hendrycks et al., 2021) tests knowledge across 57 academic subjects with multiple-choice questions. While not specifically an alignment benchmark, it measures whether alignment training preserves the model's knowledge and reasoning capabilities.

### 9.9.5 HumanEval

HumanEval (Chen et al., 2021) evaluates code generation through 164 programming problems. The pass@k metric measures the probability that at least one of $k$ generated solutions passes all test cases.

### 9.9.6 Limitations of Current Evaluation

All current evaluation methods have significant limitations:

- **LLM-as-judge biases**: LLM judges favor longer responses, responses in their own style, and responses that agree with their own "opinions" (Zheng et al., 2023).
- **Benchmark saturation**: As models improve, benchmarks become less discriminative. Scores cluster near the ceiling, making it difficult to distinguish between strong models.
- **Gaming**: Models can be specifically trained to perform well on known benchmarks without genuine improvement in general capabilities.
- **Subjectivity**: There is no ground truth for "good alignment." Different users, cultures, and contexts have different preferences.

The field increasingly recognizes that no single benchmark is sufficient. Best practice is to evaluate across multiple dimensions: capability benchmarks (MMLU, HumanEval), alignment benchmarks (MT-Bench, AlpacaEval), safety evaluations, and --- ideally --- real user feedback.

---

## 9.10 Advanced Topics

### 9.10.1 IPO: Identity Preference Optimization

Azar et al. (2023) identified a theoretical issue with DPO: it can overfit to the preference data, particularly when the data is deterministic (one response is always preferred). IPO adds a regularization term:

$$\mathcal{L}_{\text{IPO}} = \left( \log \frac{\pi_\theta(y_w|x)\pi_{\text{ref}}(y_l|x)}{\pi_\theta(y_l|x)\pi_{\text{ref}}(y_w|x)} - \frac{1}{2\beta} \right)^2$$

This squared loss prevents the implicit reward margin from growing without bound.

### 9.10.2 KTO: Kahneman-Tversky Optimization

Ethayarajh et al. (2024) proposed KTO, which works with unpaired preference data --- each example is labeled as simply "good" or "bad" without requiring paired comparisons. This dramatically simplifies data collection, as annotators only need to rate individual responses rather than compare pairs.

### 9.10.3 ORPO: Odds Ratio Preference Optimization

Hong et al. (2024) introduced ORPO, which combines SFT and preference optimization into a single training stage. The loss function adds a preference penalty to the standard language modeling loss:

$$\mathcal{L}_{\text{ORPO}} = \mathcal{L}_{\text{SFT}} + \lambda \cdot \mathcal{L}_{\text{OR}}$$

where $\mathcal{L}_{\text{OR}}$ penalizes the model when the odds ratio between chosen and rejected responses is unfavorable. This eliminates the need for a separate reference model.

---

## 9.11 Summary

The alignment pipeline transforms a raw pretrained model into one that is helpful, honest, and harmless. The field has evolved rapidly from the complex RLHF pipeline to simpler alternatives, but the core challenge remains: defining and optimizing for what humans want.

The practical recommendation for most practitioners:

1. **Start with SFT** on a high-quality, diverse instruction dataset.
2. **Apply DPO** with carefully constructed preference data ($\beta = 0.1$, learning rate $5 \times 10^{-7}$, 1 epoch).
3. **Evaluate broadly**: Use multiple benchmarks and, if possible, real user feedback.
4. **Iterate on data**: Improving the quality of instruction and preference data yields larger gains than algorithmic changes.

---

## Exercises

1. **Reward Model Training**: Train a reward model on the HH-RLHF dataset (Anthropic's Helpful and Harmless dataset). Use a small pretrained model (e.g., GPT-2 medium) with a scalar reward head. Report accuracy on held-out preferences and analyze failure cases.

2. **DPO Derivation**: Starting from the constrained RLHF optimization problem, derive the DPO loss step by step. At each step, verify the mathematical manipulations by checking dimensionality and edge cases ($\beta \to 0$, $\beta \to \infty$).

3. **DPO vs. SFT Comparison**: Fine-tune a model using (a) SFT only, (b) SFT + DPO. Compare responses qualitatively and quantitatively on a set of diverse prompts. Where does DPO improve most over SFT-only?

4. **Beta Sensitivity**: Train DPO models with $\beta \in \{0.01, 0.05, 0.1, 0.2, 0.5, 1.0\}$ and evaluate each. How does $\beta$ affect response quality, diversity, and adherence to the reference model's style?

5. **Preference Data Quality**: Create two versions of a preference dataset: one with high-quality, carefully verified preferences, and one with noisy preferences (randomly flip 20% of labels). Train DPO on both and compare. How robust is DPO to label noise?

6. **Constitutional AI Simulation**: Design a set of 10 constitutional principles. Implement a critique-revision pipeline where a model evaluates and improves its own responses. Evaluate whether the revised responses are indeed better (using a human or LLM judge).

7. **Rejection Sampling Analysis**: Implement Best-of-N sampling for $N \in \{1, 4, 16, 64, 256\}$. Plot the average reward of selected responses vs. $N$. Compare this to the reward achieved by a DPO-trained model.

---

## References

- Askell, A., et al. (2021). A General Language Assistant as a Laboratory for Alignment. *arXiv preprint arXiv:2112.00861*.
- Azar, M. G., Rowland, M., Piot, B., Guo, D., Calandriello, D., Valko, M., & Munos, R. (2023). A General Theoretical Paradigm to Understand Learning from Human Feedback. *arXiv preprint arXiv:2310.12036*.
- Bai, Y., et al. (2022). Constitutional AI: Harmlessness from AI Feedback. *arXiv preprint arXiv:2212.08073*.
- Bradley, R. A., & Terry, M. E. (1952). Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons. *Biometrika, 39*(3/4), 324--345.
- Chen, M., et al. (2021). Evaluating Large Language Models Trained on Code. *arXiv preprint arXiv:2107.03374*.
- Chung, H. W., et al. (2022). Scaling Instruction-Finetuned Language Models. *arXiv preprint arXiv:2210.11416*.
- Cui, G., et al. (2023). UltraFeedback: Boosting Language Models with High-quality Feedback. *arXiv preprint arXiv:2310.01377*.
- Ethayarajh, K., Xu, W., Muennighoff, N., Jurafsky, D., & Kiela, D. (2024). KTO: Model Alignment as Prospect Theoretic Optimization. *arXiv preprint arXiv:2402.01306*.
- Hendrycks, D., et al. (2021). Measuring Massive Multitask Language Understanding. *Proceedings of ICLR 2021*.
- Hong, J., et al. (2024). ORPO: Monolithic Preference Optimization without Reference Model. *arXiv preprint arXiv:2403.07691*.
- Li, X., et al. (2023). AlpacaEval: An Automatic Evaluator of Instruction-Following Models. *GitHub repository*.
- Ouyang, L., et al. (2022). Training Language Models to Follow Instructions with Human Feedback. *Advances in Neural Information Processing Systems, 35*, 27730--27744.
- Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. *Advances in Neural Information Processing Systems, 36*.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. *arXiv preprint arXiv:1707.06347*.
- Taori, R., et al. (2023). Stanford Alpaca: An Instruction-Following LLaMA Model. *GitHub repository*.
- Touvron, H., et al. (2023). Llama 2: Open Foundation and Fine-Tuned Chat Models. *arXiv preprint arXiv:2307.09288*.
- Wang, Y., et al. (2023). Self-Instruct: Aligning Language Models with Self-Generated Instructions. *Proceedings of ACL 2023*, 13484--13508.
- Zheng, L., et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *Advances in Neural Information Processing Systems, 36*.
- Zhou, C., et al. (2023). LIMA: Less Is More for Alignment. *Advances in Neural Information Processing Systems, 36*.
