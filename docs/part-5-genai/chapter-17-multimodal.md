# Chapter 17: Multimodal Models

## Learning Objectives

By the end of this chapter, you will be able to:

1. **Explain contrastive vision-language pretraining** — describe CLIP's architecture, training objective, and why it enabled zero-shot visual classification.
2. **Analyze generative adversarial networks** — derive the GAN minimax objective, diagnose mode collapse and training instability, and explain Wasserstein GAN and StyleGAN2 innovations.
3. **Derive the diffusion model framework** — formalize the forward and reverse processes of DDPM, connect the training objective to score matching, and distinguish DDPM from DDIM sampling.
4. **Architect Stable Diffusion** — describe how the VAE, U-Net, and CLIP text encoder interact via cross-attention, explain classifier-free guidance, and discuss ControlNet and SDXL.
5. **Build vision-language models** — explain BLIP-2's Q-Former, LLaVA's visual instruction tuning, and Flamingo's gated cross-attention.
6. **Design multimodal prompting strategies** — use GPT-4V and Claude 3.5 Vision effectively, understanding capabilities and limitations.
7. **Process documents with multimodal models** — apply LayoutLMv3 and Donut for document understanding tasks.
8. **Integrate audio, video, and unified multimodal architectures** — understand Whisper for ASR, temporal video modeling, and Gemini-style any-to-any architectures.

---

## 17.1 CLIP: Contrastive Language-Image Pretraining

### 17.1.1 Motivation and Architecture

Before CLIP, computer vision models were trained on fixed label sets — ImageNet's 1,000 classes, COCO's 80 object categories, or task-specific datasets painstakingly annotated by humans. This created two problems: models could only recognize categories they were explicitly trained on, and creating new visual classifiers required expensive labeled datasets.

CLIP (Contrastive Language-Image Pretraining; Radford et al., 2021) shattered this paradigm by learning visual concepts from natural language supervision. Trained on 400 million image-text pairs scraped from the internet, CLIP learns a shared embedding space where images and their textual descriptions are nearby, while unrelated image-text pairs are far apart.

**Dual encoder architecture.** CLIP consists of two encoders:

- **Image encoder:** Either a ResNet (modified with attention pooling) or a Vision Transformer (ViT). The ViT variant, CLIP ViT-L/14, achieved the strongest results.
- **Text encoder:** A Transformer (similar to GPT-2 architecture) that processes tokenized text and outputs a fixed-dimensional embedding from the [EOS] token position.

Both encoders project their outputs into a shared $d$-dimensional embedding space (typically $d = 512$ or $d = 768$) via learned linear projections.

### 17.1.2 Training Objective: InfoNCE

Given a batch of $N$ image-text pairs $\{(I_1, T_1), (I_2, T_2), \ldots, (I_N, T_N)\}$, CLIP computes:

- Image embeddings: $\mathbf{v}_i = f_{\text{image}}(I_i)$ for $i = 1, \ldots, N$
- Text embeddings: $\mathbf{t}_j = f_{\text{text}}(T_j)$ for $j = 1, \ldots, N$

The similarity matrix is:

$$S_{ij} = \frac{\mathbf{v}_i \cdot \mathbf{t}_j}{\|\mathbf{v}_i\| \|\mathbf{t}_j\|} \cdot e^{\tau}$$

where $\tau$ is a learned temperature parameter (log-parameterized).

The training objective is a symmetric cross-entropy loss over this $N \times N$ matrix. For image-to-text:

$$\mathcal{L}_{\text{i2t}} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(S_{ii})}{\sum_{j=1}^{N} \exp(S_{ij})}$$

And symmetrically for text-to-image:

$$\mathcal{L}_{\text{t2i}} = -\frac{1}{N} \sum_{j=1}^{N} \log \frac{\exp(S_{jj})}{\sum_{i=1}^{N} \exp(S_{ij})}$$

The total loss is:

$$\mathcal{L}_{\text{CLIP}} = \frac{1}{2}(\mathcal{L}_{\text{i2t}} + \mathcal{L}_{\text{t2i}})$$

This is equivalent to the InfoNCE loss applied symmetrically. The batch size $N$ determines the number of negatives; CLIP used $N = 32{,}768$, requiring distributed training across hundreds of GPUs.

### 17.1.3 Zero-Shot Classification

CLIP's most striking capability is zero-shot image classification. To classify an image into $C$ categories without any training:

1. Construct text prompts: "a photo of a {class name}" for each class.
2. Compute text embeddings for all $C$ prompts.
3. Compute the image embedding.
4. Predict the class with the highest cosine similarity.

```python
import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Load image
image = Image.open("photo.jpg")

# Define candidate classes
classes = ["a photo of a cat", "a photo of a dog", "a photo of a bird",
           "a photo of a car", "a photo of a building"]

# Compute similarities
inputs = processor(text=classes, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)

logits_per_image = outputs.logits_per_image  # Shape: (1, num_classes)
probs = logits_per_image.softmax(dim=1)

for cls, prob in zip(classes, probs[0]):
    print(f"{cls}: {prob:.4f}")
```

CLIP achieved competitive performance with supervised ResNet-50 on ImageNet *without ever seeing an ImageNet training image*. On broader benchmarks encompassing diverse visual domains, CLIP's zero-shot transfer outperformed many supervised models trained on specific datasets.

### 17.1.4 Why CLIP Was Revolutionary

CLIP's impact extends far beyond classification:
- **Open vocabulary:** CLIP understands any text description, not just fixed labels. It can classify images into categories that did not exist during training.
- **Foundation for generative models:** Stable Diffusion, DALL-E 2, and other text-to-image models use CLIP (or CLIP-like) text encoders to condition generation on natural language.
- **Multimodal retrieval:** CLIP embeddings enable cross-modal search — find images matching a text query, or find text descriptions matching an image.
- **Robustness:** CLIP's diverse training data makes it more robust to distribution shifts than ImageNet-trained models (Radford et al., 2021).

### 17.1.5 Linear Probe Results

CLIP's learned representations are remarkably general. A simple linear classifier (logistic regression) trained on CLIP image features achieves state-of-the-art results on many downstream tasks, often matching or exceeding models with task-specific architectures. On ImageNet, CLIP ViT-L/14 with a linear probe achieved 85.4% top-1 accuracy — competitive with the best supervised models available at the time.

---

## 17.2 GANs for Generation

### 17.2.1 The GAN Framework

Generative Adversarial Networks (Goodfellow et al., 2014) formalize image generation as a two-player game between:

- **Generator $G$:** Maps random noise $\mathbf{z} \sim p_z(\mathbf{z})$ (typically Gaussian) to synthetic images: $\mathbf{x}_{\text{fake}} = G(\mathbf{z})$.
- **Discriminator $D$:** Classifies images as real (from the training set) or fake (from the generator): $D(\mathbf{x}) \in [0, 1]$.

The two networks are trained adversarially. The generator tries to fool the discriminator; the discriminator tries to distinguish real from fake. Formally, they play a minimax game:

$$\min_G \max_D \; V(D, G) = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p_z}[\log(1 - D(G(\mathbf{z})))]$$

At the Nash equilibrium, the generator produces images indistinguishable from real data, and the discriminator outputs 0.5 for all inputs (it cannot tell the difference).

### 17.2.2 Mode Collapse and Training Instability

GAN training is notoriously difficult:

**Mode collapse** occurs when the generator learns to produce only a small subset of the possible outputs, ignoring the full diversity of the data distribution. For example, a GAN trained on faces might produce only faces of a specific gender or ethnicity.

**Training instability** manifests as oscillating losses, where the generator and discriminator alternately dominate. When the discriminator becomes too strong, the generator receives near-zero gradients and cannot learn. When the generator becomes too strong, the discriminator cannot provide useful feedback.

**Vanishing gradients** occur in the original GAN formulation when the discriminator is well-trained: $\log(1 - D(G(\mathbf{z})))$ saturates near zero, providing uninformative gradients to the generator.

### 17.2.3 Wasserstein GAN

Arjovsky et al. (2017) addressed GAN training instability by replacing the Jensen-Shannon divergence (implicit in the original GAN) with the Wasserstein distance (Earth Mover's distance):

$$W(p_{\text{data}}, p_g) = \inf_{\gamma \in \Pi(p_{\text{data}}, p_g)} \mathbb{E}_{(x, y) \sim \gamma}[\|x - y\|]$$

The WGAN objective replaces the discriminator with a **critic** $f_w$ (no sigmoid, unbounded output):

$$\min_G \max_{f_w \in \mathcal{F}} \; \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[f_w(\mathbf{x})] - \mathbb{E}_{\mathbf{z} \sim p_z}[f_w(G(\mathbf{z}))]$$

where $\mathcal{F}$ is the set of 1-Lipschitz functions. The Lipschitz constraint is enforced via weight clipping (WGAN) or gradient penalty (WGAN-GP):

$$\mathcal{L}_{\text{GP}} = \lambda \mathbb{E}_{\hat{\mathbf{x}}}[(\|\nabla_{\hat{\mathbf{x}}} f_w(\hat{\mathbf{x}})\|_2 - 1)^2]$$

where $\hat{\mathbf{x}}$ is sampled uniformly along straight lines between real and generated data points.

WGAN provides meaningful loss values that correlate with sample quality — a property the original GAN lacked — enabling practitioners to monitor training progress.

### 17.2.4 StyleGAN2

StyleGAN2 (Karras et al., 2020) represents the pinnacle of GAN-based image generation, producing photorealistic faces at 1024×1024 resolution. Its architecture introduces several innovations:

**Mapping network.** Instead of feeding the latent code $\mathbf{z}$ directly to the generator, StyleGAN first transforms it through an 8-layer MLP $f: \mathcal{Z} \to \mathcal{W}$. The intermediate latent space $\mathcal{W}$ is less entangled than $\mathcal{Z}$, meaning that each dimension of $\mathbf{w} \in \mathcal{W}$ controls a more independent aspect of the generated image.

**Synthesis network.** The generator uses a constant learned input (not noise) and progressively modulates it through **adaptive instance normalization (AdaIN)** layers controlled by the style vector $\mathbf{w}$:

$$\text{AdaIN}(\mathbf{x}_i, \mathbf{w}) = y_{s,i} \frac{\mathbf{x}_i - \mu(\mathbf{x}_i)}{\sigma(\mathbf{x}_i)} + y_{b,i}$$

where $y_s$ and $y_b$ are learned affine transformations of $\mathbf{w}$.

**Style mixing.** During training, two different latent codes $\mathbf{w}_1$ and $\mathbf{w}_2$ are used at different layers of the synthesis network. This regularizes the model and enables explicit control over generated images: coarse features (pose, face shape) are controlled by styles injected at early layers, while fine features (hair texture, skin color) are controlled by later layers.

**Path length regularization** (StyleGAN2's key improvement over StyleGAN) encourages a fixed-size step in $\mathcal{W}$ to correspond to a fixed-magnitude change in the image, improving generation quality and reducing artifacts.

---

## 17.3 Diffusion Models

### 17.3.1 Forward Process

Denoising Diffusion Probabilistic Models (Ho et al., 2020) define a forward process that gradually adds Gaussian noise to data over $T$ timesteps:

$$q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \, \mathbf{x}_{t-1}, \beta_t \mathbf{I})$$

where $\beta_1, \beta_2, \ldots, \beta_T$ is a noise schedule controlling how much noise is added at each step. As $T \to \infty$, $\mathbf{x}_T$ converges to an isotropic Gaussian $\mathcal{N}(\mathbf{0}, \mathbf{I})$.

A crucial property is that we can sample $\mathbf{x}_t$ directly from $\mathbf{x}_0$ without iterating through all intermediate steps. Define $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$:

$$q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \, \mathbf{x}_0, (1 - \bar{\alpha}_t) \mathbf{I})$$

Equivalently:

$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \, \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \, \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

This reparameterization is essential for efficient training.

### 17.3.2 Reverse Process

The reverse process learns to denoise — to recover $\mathbf{x}_{t-1}$ from $\mathbf{x}_t$:

$$p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \sigma_t^2 \mathbf{I})$$

The model $\boldsymbol{\mu}_\theta$ (a neural network parameterized by $\theta$) predicts the mean of this distribution. Ho et al. (2020) showed that reparameterizing the model to predict the noise $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ rather than the mean yields better results. The mean can be recovered as:

$$\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \right)$$

### 17.3.3 Training Objective

The DDPM training objective is derived from the variational lower bound (VLB) on the log-likelihood. After simplification, it reduces to:

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2 \right]$$

where $t \sim \text{Uniform}(1, T)$, $\mathbf{x}_0 \sim q(\mathbf{x}_0)$, and $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$. The training procedure is remarkably simple:

```python
import torch
import torch.nn as nn

class DiffusionTrainer:
    def __init__(self, model, T=1000, beta_start=1e-4, beta_end=0.02):
        self.model = model
        self.T = T

        # Linear noise schedule
        self.betas = torch.linspace(beta_start, beta_end, T)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def train_step(self, x_0: torch.Tensor) -> torch.Tensor:
        """One training step of DDPM."""
        batch_size = x_0.shape[0]

        # Sample random timesteps
        t = torch.randint(0, self.T, (batch_size,), device=x_0.device)

        # Sample noise
        epsilon = torch.randn_like(x_0)

        # Compute x_t (noisy version of x_0)
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * epsilon

        # Predict noise
        epsilon_pred = self.model(x_t, t)

        # Simple MSE loss
        loss = nn.functional.mse_loss(epsilon_pred, epsilon)
        return loss
```

### 17.3.4 Score Matching Connection

The noise prediction objective has an elegant connection to score matching. The **score function** of a distribution $p(\mathbf{x})$ is $\nabla_{\mathbf{x}} \log p(\mathbf{x})$. Song & Ermon (2019) showed that denoising score matching — training a model to predict the score of noised data — is equivalent to the DDPM objective:

$$\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \propto -\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t)$$

This connection unified the DDPM and score-based perspectives, leading to score-based generative models (Song et al., 2021) that formulate diffusion as a continuous-time stochastic differential equation.

### 17.3.5 Noise Schedule

The noise schedule $\{\beta_t\}_{t=1}^T$ significantly affects generation quality:

**Linear schedule** (Ho et al., 2020): $\beta_t$ increases linearly from $\beta_1 = 10^{-4}$ to $\beta_T = 0.02$. Simple but not optimal — noise is added too slowly at the start and too quickly at the end.

**Cosine schedule** (Nichol & Dhariwal, 2021): Designed so that $\bar{\alpha}_t$ follows a cosine curve, providing a smoother transition from clean to noisy. This produces better results, especially for low-resolution images:

$$\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)^2$$

where $s = 0.008$ is a small offset that prevents $\beta_t$ from being too small near $t = 0$.

### 17.3.6 Sampling: DDPM vs. DDIM

**DDPM sampling** is stochastic: at each step, the model predicts the noise, computes the denoised estimate, and adds fresh random noise scaled by $\sigma_t$. This requires all $T$ steps (typically $T = 1000$), making generation slow.

**DDIM (Denoising Diffusion Implicit Models)** (Song et al., 2021b) reformulates the reverse process as a deterministic mapping:

$$\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \underbrace{\left( \frac{\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \, \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{\bar{\alpha}_t}} \right)}_{\text{predicted } \mathbf{x}_0} + \sqrt{1 - \bar{\alpha}_{t-1}} \cdot \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$$

The critical advantage of DDIM is that it can skip timesteps. Instead of denoising through all $T$ steps, DDIM can use a subsequence of $S \ll T$ steps (e.g., $S = 50$ or even $S = 20$), drastically reducing generation time with minimal quality loss.

```python
@torch.no_grad()
def ddim_sample(model, shape, T=1000, steps=50, alpha_bars=None):
    """DDIM deterministic sampling."""
    device = next(model.parameters()).device

    # Create timestep subsequence
    timesteps = torch.linspace(T - 1, 0, steps, dtype=torch.long, device=device)

    # Start from pure noise
    x = torch.randn(shape, device=device)

    for i in range(len(timesteps) - 1):
        t = timesteps[i]
        t_next = timesteps[i + 1]

        # Predict noise
        eps = model(x, t.expand(shape[0]))

        # Predicted x_0
        alpha_bar_t = alpha_bars[t]
        alpha_bar_next = alpha_bars[t_next]

        x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)

        # DDIM update (deterministic)
        x = (torch.sqrt(alpha_bar_next) * x0_pred +
             torch.sqrt(1 - alpha_bar_next) * eps)

    return x
```

---

## 17.4 Stable Diffusion

### 17.4.1 Architecture Overview

Stable Diffusion (Rombach et al., 2022) made diffusion models practical by operating in a compressed **latent space** rather than pixel space. The architecture has three main components:

**VAE (Variational Autoencoder).** An image autoencoder that compresses images from pixel space to a lower-dimensional latent space and back:
- **Encoder** $\mathcal{E}$: Maps a $512 \times 512 \times 3$ image to a $64 \times 64 \times 4$ latent representation (compression factor of 48x).
- **Decoder** $\mathcal{D}$: Reconstructs the image from the latent: $\tilde{\mathbf{x}} = \mathcal{D}(\mathcal{E}(\mathbf{x}))$.

Diffusion occurs in this compressed latent space, reducing computational cost by orders of magnitude compared to pixel-space diffusion.

**U-Net.** The denoising network is a U-Net (Ronneberger et al., 2015) with several modifications:
- **Timestep conditioning:** Sinusoidal position embeddings for the diffusion timestep $t$ are injected via adaptive group normalization.
- **Cross-attention for text conditioning:** At each resolution level, cross-attention layers attend from spatial latent features to text embeddings:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d}}\right) V$$

where $Q$ comes from the latent features, and $K$, $V$ come from the text embeddings. This is the mechanism by which text controls generation.
- **Self-attention:** Spatial self-attention captures long-range dependencies within the latent.

**Text Encoder.** CLIP's text encoder (or OpenCLIP in later versions) converts the text prompt into a sequence of embeddings that condition the U-Net via cross-attention.

### 17.4.2 Classifier-Free Guidance

Classifier-free guidance (Ho & Salimans, 2022) is the key technique for generating images that strongly match the text prompt. During training, the text conditioning is randomly dropped (replaced with null embeddings) with some probability (typically 10%). At inference time, the model produces both a conditional and unconditional prediction, and guidance amplifies the difference:

$$\tilde{\boldsymbol{\epsilon}}_\theta = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, \varnothing, t) + w \cdot (\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, \mathbf{c}, t) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, \varnothing, t))$$

where $\mathbf{c}$ is the text conditioning, $\varnothing$ is the null conditioning, and $w$ is the guidance scale. Higher $w$ (typically 7–15) produces images that more closely match the prompt but with less diversity; lower $w$ produces more diverse but less prompt-faithful images.

### 17.4.3 ControlNet

ControlNet (Zhang et al., 2023) adds spatial conditioning to Stable Diffusion, enabling control over the generated image's structure (pose, edges, depth, segmentation maps) while preserving the quality of the base model.

ControlNet creates a trainable copy of the U-Net's encoder blocks. The conditioning signal (e.g., a Canny edge map or OpenPose skeleton) is processed by this copy, and the outputs are added to the original U-Net's skip connections via zero-initialized convolution layers:

$$\mathbf{y}_c = \mathcal{F}(\mathbf{x}; \Theta) + \text{zero\_conv}(\mathcal{F}(\mathbf{x} + \text{zero\_conv}(\mathbf{c}); \Theta_c))$$

The zero initialization ensures that ControlNet has no effect at the start of training, preserving the base model's generation quality while gradually learning to incorporate the spatial control signal.

### 17.4.4 SDXL Improvements

Stable Diffusion XL (Podell et al., 2023) improved upon SD 1.5 with:
- **Larger U-Net** with more attention layers and a 2048-dimensional cross-attention context.
- **Dual text encoders:** OpenCLIP ViT-bigG and CLIP ViT-L, concatenated for richer text conditioning.
- **Size and crop conditioning:** Explicit conditioning on the target image size and crop parameters, reducing artifacts from training on images of varying sizes.
- **Refinement model:** A second-stage model that operates at high resolution to add fine details.

```python
from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16"
)
pipe = pipe.to("cuda")

image = pipe(
    prompt="A serene mountain lake at sunset, oil painting style",
    negative_prompt="blurry, low quality, distorted",
    guidance_scale=7.5,
    num_inference_steps=50,
    width=1024,
    height=1024
).images[0]

image.save("mountain_lake.png")
```

---

## 17.5 BLIP-2

### 17.5.1 Architecture

BLIP-2 (Li et al., 2023) introduced a compute-efficient approach to vision-language pretraining by using a lightweight **Querying Transformer (Q-Former)** to bridge a frozen image encoder and a frozen LLM. This avoids the prohibitive cost of end-to-end training while leveraging pretrained unimodal models.

**Q-Former** is a small transformer (~188M parameters) that learns to extract visual features relevant to language understanding. It contains:
- A set of $K$ learnable **query tokens** (typically $K = 32$) that interact with image features via cross-attention.
- Self-attention layers that allow queries to interact with each other.
- Cross-attention layers that allow queries to attend to the frozen image encoder's output features.

### 17.5.2 Two-Stage Training

**Stage 1: Representation Learning.** The Q-Former is trained on three objectives with a frozen image encoder:
- **Image-Text Contrastive (ITC):** Align the Q-Former's output with the text encoder's output, similar to CLIP.
- **Image-grounded Text Generation (ITG):** Generate text conditioned on image features extracted by the Q-Former.
- **Image-Text Matching (ITM):** Binary classification of whether an image-text pair matches.

**Stage 2: Generative Learning.** The Q-Former's output is projected into the LLM's embedding space via a linear layer. The LLM (frozen) generates text conditioned on the visual tokens:

$$\text{output} = \text{LLM}([\text{Q-Former}(I); \text{TextTokens}])$$

This two-stage approach allows BLIP-2 to leverage any frozen LLM (OPT, FlanT5) without modifying its weights, making it extremely parameter-efficient.

```python
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16, device_map="auto"
)

image = Image.open("photo.jpg")

# Image captioning
inputs = processor(images=image, return_tensors="pt").to("cuda", torch.float16)
generated_ids = model.generate(**inputs, max_new_tokens=50)
caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(f"Caption: {caption}")

# Visual question answering
prompt = "Question: What color is the sky in this image? Answer:"
inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda", torch.float16)
generated_ids = model.generate(**inputs, max_new_tokens=20)
answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(f"Answer: {answer}")
```

---

## 17.6 LLaVA

### 17.6.1 Visual Instruction Tuning

LLaVA (Liu et al., 2023) demonstrated that connecting a visual encoder to an LLM with a simple linear projection, followed by instruction tuning, produces a powerful multimodal assistant. Its simplicity was its greatest innovation.

**Architecture:**
1. A frozen CLIP visual encoder (ViT-L/14) extracts image features.
2. A trainable **projection layer** (linear or MLP) maps CLIP features into the LLM's embedding space.
3. A pretrained LLM (Vicuna, LLaMA) processes the combined visual and text tokens.

The visual tokens and text tokens are concatenated and processed by the LLM as a single sequence:

$$\mathbf{H}_v = W \cdot f_{\text{CLIP}}(I), \quad \text{tokens} = [\mathbf{H}_v; \text{TextTokens}]$$

### 17.6.2 Training Process

**Stage 1: Pretrain projection.** The projection layer is trained on image-caption pairs (595K samples from CC3M), with both the visual encoder and LLM frozen. This aligns the visual feature space with the language model's input space.

**Stage 2: Instruction tuning.** The projection layer and LLM are fine-tuned on multimodal instruction-following data (158K conversations generated by GPT-4). The visual encoder remains frozen. This teaches the model to follow complex visual instructions — describe images in detail, answer questions about visual content, reason about spatial relationships, and more.

```python
# Conceptual LLaVA forward pass
class LLaVA(nn.Module):
    def __init__(self, clip_model, projection, llm):
        super().__init__()
        self.visual_encoder = clip_model  # Frozen
        self.projection = projection       # Trainable
        self.llm = llm                     # Trainable in stage 2

    def forward(self, image, text_tokens):
        # Extract visual features
        with torch.no_grad():
            visual_features = self.visual_encoder.encode_image(image)

        # Project to LLM embedding space
        visual_tokens = self.projection(visual_features)  # (B, N_v, D_llm)

        # Get text embeddings
        text_embeds = self.llm.get_input_embeddings()(text_tokens)

        # Concatenate visual and text tokens
        inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1)

        # Generate
        outputs = self.llm(inputs_embeds=inputs_embeds)
        return outputs
```

LLaVA's impact was outsized relative to its simplicity. It showed that sophisticated multimodal reasoning emerges from combining strong unimodal models with relatively simple integration — the visual instruction-tuning data, not the architecture, was the key ingredient.

---

## 17.7 GPT-4V and Claude 3.5 Vision

### 17.7.1 Capabilities

GPT-4V (OpenAI, 2023) and Claude 3.5 Sonnet (Anthropic, 2024) represent the frontier of multimodal language models. Their capabilities include:

- **Detailed image understanding:** Describing complex scenes, identifying objects, reading text in images (OCR), understanding charts and diagrams.
- **Spatial reasoning:** Understanding relative positions, counting objects, interpreting maps.
- **Multi-image reasoning:** Comparing multiple images, tracking changes over time.
- **Document analysis:** Processing scanned documents, forms, receipts, and handwritten text.
- **Code from screenshots:** Generating HTML/CSS from website screenshots or code from flowcharts.

### 17.7.2 Multimodal Prompting Strategies

Effective multimodal prompting requires understanding how these models process visual input:

```python
from openai import OpenAI
import base64

client = OpenAI()

def analyze_image(image_path: str, question: str) -> str:
    """Analyze an image using GPT-4V with structured prompting."""
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a precise visual analyst. Describe what you see "
                           "accurately. If uncertain, express your uncertainty."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}",
                            "detail": "high"  # "low", "high", or "auto"
                        }
                    }
                ]
            }
        ],
        max_tokens=1000
    )
    return response.choices[0].message.content
```

**Best practices for multimodal prompting:**
- Use `detail: "high"` for images requiring fine-grained analysis (documents, small text, detailed diagrams).
- Reference specific regions of the image in your prompt ("In the top-left corner...").
- For multi-image tasks, clearly label which image you are referring to.
- Ask for structured output (JSON, tables) when extracting specific information.
- Chain visual analysis with reasoning ("First describe what you see, then analyze what it means").

### 17.7.3 Limitations

Current multimodal models have known limitations:
- **Counting:** Reliably counting objects (especially > 10) remains difficult.
- **Spatial precision:** Exact coordinates and measurements are unreliable.
- **Hallucinated text:** Models may "read" text that is not actually in the image.
- **Complex diagrams:** Highly technical or densely packed visualizations may be misinterpreted.
- **Bias:** Models inherit biases from training data regarding demographics, cultural representations, and stereotypes.

---

## 17.8 Flamingo and Idefics

### 17.8.1 Flamingo Architecture

Flamingo (Alayrac et al., 2022) from DeepMind was among the first models to demonstrate strong few-shot multimodal learning. Its architecture introduces two key innovations:

**Perceiver Resampler.** Converts variable-length visual feature sequences into a fixed-length set of visual tokens. A set of learnable latent queries attend to the image features via cross-attention, producing a compact visual representation regardless of image resolution or aspect ratio.

$$\mathbf{Z} = \text{PerceiverResampler}(f_{\text{visual}}(I)) \in \mathbb{R}^{N_q \times D}$$

where $N_q$ is the fixed number of query tokens (typically 64).

**Gated cross-attention.** Visual information is injected into a frozen LLM via gated cross-attention layers inserted between existing transformer layers. The gating mechanism starts at zero, ensuring the model initially behaves exactly like the frozen LLM:

$$\mathbf{h}' = \mathbf{h} + \tanh(\alpha) \cdot \text{CrossAttention}(\mathbf{h}, \mathbf{Z})$$

where $\alpha$ is a learnable scalar initialized to zero.

### 17.8.2 Few-Shot Multimodal Learning

Flamingo's architecture enables few-shot multimodal learning: given a few image-text examples in the prompt, the model can perform new tasks without any gradient updates. For example, providing 2–3 examples of image-to-caption pairs in the prompt enables Flamingo to caption new images in the same style.

**Idefics** (Laurençon et al., 2023) is an open-source reproduction of Flamingo, built on LLaMA and trained on publicly available datasets. Idefics2 improved upon the original with better training data curation and architectural refinements.

---

## 17.9 Document Understanding

### 17.9.1 LayoutLMv3

LayoutLMv3 (Huang et al., 2022) processes documents by combining three modalities:

- **Text:** OCR-extracted text tokens with their spatial positions.
- **Layout:** 2D position embeddings encoding the bounding box coordinates $(x_0, y_0, x_1, y_1)$ of each text segment.
- **Image:** Patch embeddings from the document image.

All three modalities are projected into a shared embedding space and processed by a single transformer. The key innovation is **unified pretraining** — LayoutLMv3 uses masked language modeling, masked image modeling, and word-patch alignment objectives simultaneously during pretraining.

```python
from transformers import AutoProcessor, AutoModelForTokenClassification
from PIL import Image

# Document understanding with LayoutLMv3
processor = AutoProcessor.from_pretrained(
    "microsoft/layoutlmv3-base", apply_ocr=True
)
model = AutoModelForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base", num_labels=7  # e.g., for form field types
)

image = Image.open("invoice.png")
encoding = processor(image, return_tensors="pt")

outputs = model(**encoding)
predictions = outputs.logits.argmax(-1).squeeze().tolist()
```

### 17.9.2 Donut: OCR-Free Document Understanding

Donut (Kim et al., 2022) eliminates the OCR dependency entirely. It uses a Swin Transformer as a visual encoder and a BART-style decoder that directly generates structured output from the document image:

$$\text{output} = \text{Decoder}(\text{SwinEncoder}(\text{image}), \text{prompt})$$

Donut can perform document classification, information extraction, and visual question answering without any OCR preprocessing, making it more robust to OCR errors and simpler to deploy.

### 17.9.3 Practical Applications

Document understanding models power:
- **Invoice processing:** Extract vendor, amount, date, line items.
- **Form parsing:** Identify fields and their values in insurance forms, tax documents, medical records.
- **Receipt analysis:** Extract merchant, total, items, and payment method.
- **Contract analysis:** Identify parties, dates, clauses, and obligations.

---

## 17.10 Video Understanding

### 17.10.1 Temporal Modeling Approaches

Video understanding extends image understanding with temporal reasoning. Key approaches include:

**Frame sampling.** Sample $K$ frames uniformly from the video and process each independently with an image model. Simple but loses temporal information.

**Temporal attention.** Add temporal attention layers between spatial attention layers in a vision transformer. Each frame attends to other frames, capturing motion and temporal dynamics.

**3D convolutions.** Extend 2D convolutional networks to 3D (space + time). C3D and SlowFast networks use different temporal resolutions to capture both fast motion and slow scene changes.

### 17.10.2 VideoLLaMA and Video-LLMs

VideoLLaMA (Zhang et al., 2023b) extends the LLaVA-style architecture to video:

1. Sample frames from the video at regular intervals.
2. Extract visual features from each frame using a visual encoder (CLIP, EVA-CLIP).
3. Use a temporal aggregation module (e.g., Q-Former with temporal position embeddings) to compress the frame-level features into a fixed-length representation.
4. Project the aggregated video features into the LLM's embedding space.
5. The LLM generates text conditioned on the video features and the user's question.

### 17.10.3 Video-Text Alignment

Aligning video and text is more challenging than image-text alignment because:
- Videos contain temporal dynamics that static text struggles to describe.
- The same video can be described at different temporal granularities (frame-by-frame narration vs. high-level summary).
- Annotation is expensive — video captioning datasets are smaller than image captioning datasets.

Recent approaches use ASR (automatic speech recognition) transcripts as a free source of video-text alignment, training on instructional videos (HowTo100M) or web videos with subtitles.

---

## 17.11 Audio

### 17.11.1 Whisper

Whisper (Radford et al., 2023) is an encoder-decoder transformer for automatic speech recognition (ASR) that achieves robust, multilingual speech recognition through large-scale weak supervision.

**Architecture:**
- **Encoder:** Processes log-Mel spectrograms (80-channel, 30-second windows) through convolutional layers followed by transformer blocks with sinusoidal position embeddings.
- **Decoder:** An autoregressive transformer that generates text tokens conditioned on the encoder output and special task tokens.

**Task tokens** condition the model on what to do:
- `<|startoftranscript|>` followed by a language token (e.g., `<|en|>`)
- `<|transcribe|>` for transcription or `<|translate|>` for translation to English
- `<|timestamps|>` for word-level timestamp generation

```python
import whisper

model = whisper.load_model("large-v3")

# Transcribe audio
result = model.transcribe(
    "audio.mp3",
    language="en",
    task="transcribe",
    word_timestamps=True
)

print(result["text"])
for segment in result["segments"]:
    print(f"[{segment['start']:.2f} - {segment['end']:.2f}] {segment['text']}")
```

**Training data:** Whisper was trained on 680,000 hours of multilingual audio-text pairs from the internet. This weak supervision (imperfect transcriptions) proved sufficient for state-of-the-art quality because the sheer volume of data overwhelmed the noise.

**Robustness:** Whisper generalizes remarkably well to diverse accents, background noise, and domain-specific vocabulary without fine-tuning, thanks to its diverse training data.

### 17.11.2 MusicGen

MusicGen (Copet et al., 2023) generates high-quality music from text descriptions or melodic conditioning. It uses a single-stage autoregressive transformer over compressed audio tokens (from EnCodec), avoiding the complexity of multi-stage cascade models.

Key innovations include:
- **Codebook interleaving patterns:** Instead of generating all codebook levels autoregressively (which would be very slow), MusicGen uses efficient interleaving patterns that allow parallel generation across codebook levels.
- **Text conditioning:** CLIP-like text embeddings condition the generation via cross-attention.
- **Melody conditioning:** An optional Chroma feature extractor provides melodic conditioning, allowing the model to generate music that follows a given melody while varying instrumentation and style.

### 17.11.3 AudioLM

AudioLM (Borsos et al., 2023) generates realistic audio continuations — speech, music, or environmental sounds — by operating on multiple levels of audio tokenization:
- **Semantic tokens** (from w2v-BERT) capture high-level content.
- **Acoustic tokens** (from SoundStream) capture fine-grained acoustic details.

Generation proceeds hierarchically: first generate semantic tokens (capturing "what" to say or play), then generate acoustic tokens conditioned on the semantics (capturing "how" it sounds). This hierarchical approach produces coherent, natural-sounding audio.

---

## 17.12 Unified Multimodal Architectures

### 17.12.1 Gemini-Style Any-to-Any

The frontier of multimodal AI is **any-to-any** architectures — models that can accept and generate any combination of modalities (text, images, audio, video, code) within a single model. Google's Gemini (Team et al., 2023) exemplifies this approach.

**Tokenizing different modalities.** The key challenge is representing all modalities in a common token space:

- **Text:** Standard BPE tokenization.
- **Images:** Either patch-based tokenization (ViT-style) or VQ-VAE tokenization into discrete tokens.
- **Audio:** Spectral tokenization or neural audio codec tokens (EnCodec, SoundStream).
- **Video:** Temporal frame sampling + spatial tokenization, possibly with temporal compression.

```
Input:  [TEXT_TOKENS] [IMAGE_TOKENS] [AUDIO_TOKENS]
         ↓              ↓              ↓
    Text Tokenizer  Image Tokenizer  Audio Tokenizer
         ↓              ↓              ↓
    ┌─────────────────────────────────────────┐
    │        Unified Transformer Backbone      │
    │   (shared attention over all modalities) │
    └─────────────────────────────────────────┘
         ↓              ↓              ↓
    Text Decoder    Image Decoder   Audio Decoder
         ↓              ↓              ↓
Output: [TEXT]       [IMAGE]        [AUDIO]
```

### 17.12.2 Challenges

Building unified multimodal models presents unique challenges:

**Modality imbalance.** Text data is orders of magnitude more abundant than paired multimodal data. Training must balance learning across modalities without the dominant modality overwhelming the others.

**Resolution-computation tradeoff.** High-resolution images and long videos consume enormous numbers of tokens. A single 1024×1024 image at 16×16 patch size produces 4,096 tokens — more than many text prompts. Efficient tokenization and compression are essential.

**Evaluation.** Evaluating any-to-any models requires benchmarks that test cross-modal reasoning, not just unimodal quality. A model might generate excellent text and excellent images independently but fail at generating an image that accurately reflects a complex text description.

**Alignment.** Ensuring that the model's understanding is consistent across modalities — that its "concept" of a cat is the same whether encountered in text, image, or audio — requires careful pretraining on aligned multimodal data.

### 17.12.3 Toward General-Purpose Multimodal Agents

The convergence of multimodal understanding and agentic capabilities points toward general-purpose assistants that can:
- See (process images and video)
- Hear (process audio and speech)
- Read (process text and documents)
- Speak (generate speech)
- Create (generate images and video)
- Act (use tools, interact with software)

Models like GPT-4o (omni), Gemini Ultra, and Claude 3.5 represent early steps toward this vision, though significant gaps remain in generation quality, consistency across modalities, and real-time performance.

---

## Summary

Multimodal AI has undergone a remarkable transformation. From the foundational work of GANs and the paradigm shift of CLIP, through the revolution of diffusion models and the emergence of vision-language models, we have arrived at an era where a single model can understand and generate across multiple modalities.

This chapter covered the theoretical foundations (GAN training, diffusion process derivation, contrastive learning), practical architectures (Stable Diffusion, BLIP-2, LLaVA, Whisper), and frontier directions (unified multimodal models, any-to-any generation). The key themes are:

1. **Contrastive learning** bridges modalities by learning shared embedding spaces.
2. **Diffusion models** have largely replaced GANs for image generation, offering superior training stability and quality.
3. **Modular architectures** (frozen encoders + trainable adapters) enable efficient multimodal model development.
4. **Scale** — in both data and model size — consistently unlocks new capabilities.
5. **Unification** — processing all modalities in a single model — is the emerging frontier.

---

## Exercises

1. **CLIP zero-shot classifier.** Build a zero-shot image classifier using CLIP that categorizes images into 20 custom categories of your choice. Compare its accuracy to a ResNet-50 fine-tuned on a small labeled dataset (100 images per category).

2. **Prompt engineering for Stable Diffusion.** Generate images with Stable Diffusion using 10 different prompts. For each, experiment with guidance scale (3, 7, 12, 20) and document how it affects prompt adherence and image quality.

3. **Diffusion from scratch.** Implement a simple DDPM on the MNIST dataset. Train for 100 epochs and visualize generated digits. Compare DDPM and DDIM sampling at 1000, 100, and 20 steps.

4. **BLIP-2 evaluation.** Evaluate BLIP-2's image captioning quality on 50 diverse images using BLEU, METEOR, and CIDEr metrics. Identify categories of images where BLIP-2 performs best and worst.

5. **LLaVA visual reasoning.** Test LLaVA or GPT-4V on 20 visual reasoning tasks — chart reading, spatial reasoning, document understanding, multi-image comparison — and categorize the types of errors each model makes.

6. **Multimodal RAG.** Build a multimodal RAG system that indexes both text and images from a set of technical documents. Use CLIP for image retrieval and a text embedder for text retrieval. Evaluate whether including images in the retrieved context improves answer quality.

7. **Audio transcription pipeline.** Build an end-to-end audio processing pipeline: (a) transcribe with Whisper, (b) diarize speakers (identify who is speaking), (c) summarize the transcript with an LLM. Test on a 30-minute meeting recording.

---

## References

Alayrac, J. B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., ... & Simonyan, K. (2022). Flamingo: A visual language model for few-shot learning. *Advances in Neural Information Processing Systems*, 35, 23716–23736.

Arjovsky, M., Chintala, S., & Bottou, L. (2017). Wasserstein generative adversarial networks. *Proceedings of the 34th International Conference on Machine Learning*, 214–223.

Borsos, Z., Marinier, R., Vincent, D., Kharitonov, E., Pietquin, O., Sharifi, M., ... & Tagliasacchi, A. (2023). AudioLM: A language modeling approach to audio generation. *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, 31, 2523–2533.

Copet, J., Kreuk, F., Gat, I., Remez, T., Kant, D., Synnaeve, G., ... & Défossez, A. (2023). Simple and controllable music generation. *Advances in Neural Information Processing Systems*, 36.

Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ... & Bengio, Y. (2014). Generative adversarial nets. *Advances in Neural Information Processing Systems*, 27.

Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. *Advances in Neural Information Processing Systems*, 33, 6840–6851.

Ho, J., & Salimans, T. (2022). Classifier-free diffusion guidance. *arXiv preprint arXiv:2207.12598*.

Huang, Y., Lv, T., Cui, L., Lu, Y., & Wei, F. (2022). LayoutLMv3: Pre-training for document AI with unified text and image masking. *Proceedings of the 30th ACM International Conference on Multimedia*, 4083–4091.

Karras, T., Laine, S., Aittala, M., Hellsten, J., Lehtinen, J., & Aila, T. (2020). Analyzing and improving the image quality of StyleGAN. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 8110–8119.

Kim, G., Hong, T., Yim, M., Nam, J., Park, J., Yim, J., ... & Park, S. (2022). OCR-free document understanding transformer. *European Conference on Computer Vision*, 498–517.

Laurençon, H., Tronchon, B., Cha, M., Pari, S., Singh, A., Saulnier, L., ... & Scialom, T. (2023). OBELICS: An open web-scale filtered dataset of interleaved image-text documents. *Advances in Neural Information Processing Systems*, 36.

Li, J., Li, D., Savarese, S., & Hoi, S. (2023). BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. *Proceedings of the 40th International Conference on Machine Learning*, 19730–19742.

Liu, H., Li, C., Wu, Q., & Lee, Y. J. (2023). Visual instruction tuning. *Advances in Neural Information Processing Systems*, 36.

Nichol, A. Q., & Dhariwal, P. (2021). Improved denoising diffusion probabilistic models. *Proceedings of the 38th International Conference on Machine Learning*, 8162–8171.

OpenAI. (2023). GPT-4V(ision) system card. *OpenAI Technical Report*.

Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., Müller, J., ... & Rombach, R. (2023). SDXL: Improving latent diffusion models for high-resolution image synthesis. *arXiv preprint arXiv:2307.01952*.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., ... & Sutskever, I. (2021). Learning transferable visual models from natural language supervision. *Proceedings of the 38th International Conference on Machine Learning*, 8748–8763.

Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2023). Robust speech recognition via large-scale weak supervision. *Proceedings of the 40th International Conference on Machine Learning*, 28492–28518.

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). High-resolution image synthesis with latent diffusion models. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 10684–10695.

Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. *Medical Image Computing and Computer-Assisted Intervention*, 234–241.

Song, Y., & Ermon, S. (2019). Generative modeling by estimating gradients of the data distribution. *Advances in Neural Information Processing Systems*, 32.

Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021). Score-based generative modeling through stochastic differential equations. *Proceedings of the International Conference on Learning Representations*.

Song, J., Meng, C., & Ermon, S. (2021b). Denoising diffusion implicit models. *Proceedings of the International Conference on Learning Representations*.

Team, G., Anil, R., Borgeaud, S., Wu, Y., Alayrac, J. B., Yu, J., ... & Hassabis, D. (2023). Gemini: A family of highly capable multimodal models. *arXiv preprint arXiv:2312.11805*.

Zhang, L., Rao, A., & Agrawala, M. (2023). Adding conditional control to text-to-image diffusion models. *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 3836–3847.

Zhang, H., Li, X., & Bing, L. (2023b). Video-LLaMA: An instruction-tuned audio-visual language model for video understanding. *arXiv preprint arXiv:2306.02858*.
