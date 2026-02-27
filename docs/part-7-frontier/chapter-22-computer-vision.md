# Chapter 22: Advanced Computer Vision

> *"The question is no longer whether machines can see — it is whether they can see as richly, as contextually, and as creatively as humans do."*

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Explain the architecture of Vision Transformers (ViT, DeiT, Swin) and articulate why they have supplanted CNNs for many tasks.
2. Trace the evolution of object detection from R-CNN through YOLO and DETR, implementing modern detectors for real-world applications.
3. Distinguish between semantic, instance, and panoptic segmentation, and deploy the Segment Anything Model (SAM) for promptable segmentation.
4. Understand neural radiance fields (NeRF) and 3D Gaussian Splatting for novel view synthesis, and process point clouds with PointNet architectures.
5. Build video understanding pipelines using temporal modeling, video transformers, and multimodal video-language models.
6. Apply pose estimation models (MediaPipe, OpenPose, ViTPose) for human body, hand, and face landmark detection.
7. Deploy OCR pipelines using PaddleOCR, TrOCR, and document AI systems at production scale.
8. Implement monocular and stereo depth estimation using MiDaS, DPT, and multi-view geometry methods.

---

## 22.1 Vision Transformers

The arrival of Vision Transformers marked a paradigm shift in computer vision. For decades, convolutional neural networks (CNNs) dominated — from LeNet to AlexNet to ResNet. The inductive biases of convolutions (locality, translation equivariance) seemed perfectly suited to visual data. Then Dosovitskiy et al. (2021) asked a provocative question: *what if we just applied a standard Transformer, with minimal modifications, directly to images?*

### 22.1.1 ViT: Vision Transformer

The Vision Transformer (ViT) applies the Transformer architecture — originally designed for natural language processing — directly to sequences of image patches (Dosovitskiy et al., 2021).

**Patch Embedding.** An image $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$ is reshaped into a sequence of flattened 2D patches $\mathbf{x}_p \in \mathbb{R}^{N \times (P^2 \cdot C)}$, where $(H, W)$ is the resolution, $C$ is the number of channels, $P$ is the patch size, and $N = HW/P^2$ is the number of patches. Each patch is linearly projected to a $D$-dimensional embedding:

$$\mathbf{z}_0 = [\mathbf{x}_\text{class}; \, \mathbf{x}_p^1 \mathbf{E}; \, \mathbf{x}_p^2 \mathbf{E}; \, \cdots; \, \mathbf{x}_p^N \mathbf{E}] + \mathbf{E}_\text{pos}$$

where $\mathbf{E} \in \mathbb{R}^{(P^2 \cdot C) \times D}$ is the patch embedding projection and $\mathbf{E}_\text{pos} \in \mathbb{R}^{(N+1) \times D}$ is the positional embedding.

**Class Token.** A learnable embedding $\mathbf{x}_\text{class}$ is prepended to the sequence. After passing through Transformer layers, the state at this token serves as the image representation for classification — analogous to BERT's `[CLS]` token.

**Positional Encoding.** Unlike the fixed sinusoidal encodings used in the original Transformer, ViT uses learnable 1D positional embeddings. These are added to the patch embeddings and learned during training. Remarkably, learned 2D-aware positional embeddings emerge — nearby patches develop similar positional embeddings, and the grid structure is recovered automatically.

**Transformer Encoder.** The sequence of embedded patches is processed by a standard Transformer encoder consisting of $L$ layers of multi-head self-attention (MSA) and MLP blocks with LayerNorm (LN) and residual connections:

$$\mathbf{z}'_\ell = \text{MSA}(\text{LN}(\mathbf{z}_{\ell-1})) + \mathbf{z}_{\ell-1}$$
$$\mathbf{z}_\ell = \text{MLP}(\text{LN}(\mathbf{z}'_\ell)) + \mathbf{z}'_\ell$$

The final classification head reads the class token:

$$\mathbf{y} = \text{LN}(\mathbf{z}_L^0)$$

**Comparison to CNNs.** ViT has fundamentally different inductive biases from CNNs:

| Property | CNN | ViT |
|----------|-----|-----|
| Locality | Built-in (conv kernel) | Learned (self-attention is global) |
| Translation equivariance | Built-in (weight sharing) | Must be learned from data |
| Data efficiency | Better with small data | Requires large-scale pretraining |
| Scalability | Saturates at extreme scale | Performance scales with data and compute |
| Receptive field | Grows with depth | Global from layer 1 |

ViT underperforms CNNs when trained on mid-size datasets (e.g., ImageNet-1k alone) because it lacks the strong inductive biases that CNNs enjoy. However, when pretrained on large datasets (ImageNet-21k, JFT-300M), ViT surpasses CNNs, suggesting that inductive biases are helpful primarily when data is limited.

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """Convert image into patch embeddings."""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        # Linear projection of flattened patches (equivalent to Conv2d)
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, H, W) -> (B, embed_dim, H/P, W/P) -> (B, num_patches, embed_dim)
        x = self.projection(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class ViT(nn.Module):
    """Simplified Vision Transformer."""
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 num_classes=1000, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4.0, drop_rate=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=drop_rate,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm (as in ViT)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, N, D)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, D)
        x = self.pos_drop(x + self.pos_embed)

        x = self.transformer(x)
        x = self.norm(x[:, 0])  # CLS token output
        return self.head(x)

# Example usage
model = ViT(img_size=224, patch_size=16, num_classes=1000)
img = torch.randn(2, 3, 224, 224)
logits = model(img)  # (2, 1000)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### 22.1.2 DeiT: Data-Efficient Image Transformers

A major limitation of ViT is its hunger for data — the original paper required pretraining on JFT-300M (300 million images). Touvron et al. (2021) introduced DeiT (Data-efficient Image Transformers), which matches ViT performance while training only on ImageNet-1k (1.2 million images).

**Knowledge Distillation from CNN to ViT.** DeiT introduces a *distillation token* — a second learnable token appended to the patch sequence alongside the class token. During training:

- The class token is trained with cross-entropy against the true labels (hard labels).
- The distillation token is trained to mimic the output of a strong CNN teacher (e.g., RegNetY-16GF).

The distillation loss uses *hard distillation* — the student is trained on the teacher's argmax predictions rather than its soft probability distribution, which surprisingly works better:

$$\mathcal{L} = (1 - \lambda) \mathcal{L}_\text{CE}(y, \psi(\mathbf{z}_\text{cls})) + \lambda \mathcal{L}_\text{CE}(y_t, \psi(\mathbf{z}_\text{dist}))$$

where $y_t = \arg\max_c f_\text{teacher}(x)_c$ is the teacher's hard prediction.

At inference, the class and distillation token outputs are averaged. This CNN-to-ViT distillation is particularly effective because the teacher's convolutional inductive biases guide the student's learning, compensating for ViT's lack of built-in locality.

**Key DeiT Training Recipes.** Beyond distillation, DeiT introduced a comprehensive training recipe including strong data augmentation (RandAugment, Mixup, CutMix), regularization (stochastic depth, repeated augmentation), and careful hyperparameter selection — demonstrating that training methodology matters as much as architecture.

### 22.1.3 Swin Transformer

While ViT computes global self-attention over all patches (quadratic in sequence length), the Swin Transformer (Liu et al., 2021) introduces a hierarchical vision transformer that computes self-attention within local windows, achieving linear computational complexity with respect to image size.

**Shifted Windows.** The core innovation is *shifted window multi-head self-attention* (SW-MSA):

1. **Window Partition:** The feature map is partitioned into non-overlapping windows of size $M \times M$ patches (typically $M = 7$). Self-attention is computed independently within each window.

2. **Shifted Windows:** In alternating layers, the window partition is shifted by $(\lfloor M/2 \rfloor, \lfloor M/2 \rfloor)$ pixels, creating cross-window connections. An efficient batch computation with masking handles boundary conditions.

This alternation between regular and shifted window partitioning provides cross-window connectivity while maintaining linear complexity:

$$\Omega(\text{W-MSA}) = 4hwC^2 + 2M^2hwC$$
$$\Omega(\text{MSA}) = 4hwC^2 + 2(hw)^2C$$

where $hw$ is the number of patches and $C$ is the embedding dimension.

**Hierarchical Representation.** Unlike ViT's single-resolution feature map, Swin produces a hierarchical pyramid of features — similar to ResNet — by merging patches at each stage:

- **Stage 1:** $\frac{H}{4} \times \frac{W}{4}$ patches with $C$ channels
- **Stage 2:** $\frac{H}{8} \times \frac{W}{8}$ patches with $2C$ channels (after patch merging)
- **Stage 3:** $\frac{H}{16} \times \frac{W}{16}$ patches with $4C$ channels
- **Stage 4:** $\frac{H}{32} \times \frac{W}{32}$ patches with $8C$ channels

This hierarchical design makes Swin a general-purpose backbone for dense prediction tasks like detection and segmentation, unlike ViT which only produces single-scale features. Swin Transformer achieves state-of-the-art results on COCO object detection, ADE20K semantic segmentation, and ImageNet classification, establishing transformers as a viable replacement for CNNs across all vision tasks.

---

## 22.2 Object Detection

Object detection — localizing and classifying objects in images — has been one of the most actively researched areas in computer vision. The field has evolved from slow, multi-stage pipelines to fast, end-to-end systems.

### 22.2.1 The R-CNN Family

**R-CNN (Girshick et al., 2014).** The pioneering deep learning detector used selective search to propose ~2,000 region candidates, warped each to a fixed size, extracted CNN features independently for each region, then classified with SVMs and refined bounding boxes with regression. Accurate but painfully slow (~47 seconds per image).

**Fast R-CNN (Girshick, 2015).** Rather than processing each region independently, the entire image is passed through a CNN once. Region proposals are projected onto the feature map, and a *RoI Pooling* layer extracts fixed-size features for each proposal. Multi-task training jointly predicts class probabilities and bounding box offsets. This reduced inference to ~0.3 seconds per image.

**Faster R-CNN (Ren et al., 2015).** Replaced external region proposal methods with a *Region Proposal Network* (RPN) that shares convolutional features with the detection network. The RPN uses anchor boxes at multiple scales and aspect ratios to predict objectness and refine boxes. This made the proposal step nearly cost-free, achieving ~0.2 seconds per image (5 fps).

The two-stage pipeline is:
1. **RPN** generates proposals (class-agnostic bounding boxes).
2. **Detection head** classifies each proposal and refines its bounding box.

The loss function combines classification and regression:

$$\mathcal{L} = \frac{1}{N_\text{cls}} \sum_i \mathcal{L}_\text{cls}(p_i, p_i^*) + \lambda \frac{1}{N_\text{reg}} \sum_i p_i^* \mathcal{L}_\text{reg}(t_i, t_i^*)$$

where $p_i^*$ is the ground-truth objectness label and $t_i, t_i^*$ are predicted and ground-truth box regression targets.

### 22.2.2 YOLO: You Only Look Once

YOLO (Redmon et al., 2016) introduced single-stage detection — predicting bounding boxes and class probabilities directly from full images in one evaluation, without a separate proposal stage.

**Core Idea.** The image is divided into an $S \times S$ grid. Each grid cell predicts $B$ bounding boxes (center, width, height, confidence) and $C$ class probabilities. Non-Maximum Suppression (NMS) removes duplicate detections. This single-shot approach trades some accuracy for dramatic speed gains.

**YOLOv5 and YOLOv8.** Modern YOLO variants (Jocher, 2023) have evolved far beyond the original design:

- **YOLOv5** introduced a clean PyTorch implementation with CSPNet backbone, PANet feature pyramid, and extensive data augmentation (mosaic, mixup). It defined model scales (n, s, m, l, x) for different speed/accuracy tradeoffs.

- **YOLOv8** is the current state-of-the-art in the YOLO family, featuring:
  - **Anchor-free detection:** Eliminates predefined anchor boxes. Instead, each cell directly predicts the center offset and width/height. This removes the need for anchor tuning and simplifies the pipeline.
  - **Decoupled head:** Separate branches for classification and regression (instead of a shared head), improving both tasks.
  - **Distribution Focal Loss (DFL):** Models bounding box coordinates as discrete probability distributions rather than single regression values, improving localization.
  - **Task-specific heads:** Unified architecture supports detection, segmentation, classification, and pose estimation.

```python
from ultralytics import YOLO

# Load a pretrained YOLOv8 model
model = YOLO("yolov8n.pt")  # nano variant for speed

# Inference on an image
results = model("image.jpg")

# Access detections
for result in results:
    boxes = result.boxes
    for box in boxes:
        xyxy = box.xyxy[0].tolist()    # [x1, y1, x2, y2]
        confidence = box.conf[0].item()
        class_id = int(box.cls[0].item())
        class_name = model.names[class_id]
        print(f"{class_name}: {confidence:.2f} at {xyxy}")

# Fine-tune on custom dataset
model.train(data="custom_dataset.yaml", epochs=100, imgsz=640, batch=16)

# Export for deployment
model.export(format="onnx")
```

### 22.2.3 DETR: Detection Transformer

DETR (DEtection TRansformer) by Carion et al. (2020) fundamentally reimagines object detection as a *set prediction* problem, eliminating anchors, NMS, and many hand-designed components.

**Architecture.** DETR consists of:
1. A CNN backbone extracts features from the image.
2. A Transformer encoder processes the flattened feature map with positional encoding.
3. A Transformer decoder takes $N$ learnable *object queries* (e.g., $N = 100$) and attends to the encoder output, producing $N$ prediction slots.
4. Each prediction slot outputs either a (class, bounding box) pair or $\varnothing$ (no object).

**Set Prediction and Hungarian Matching.** The key insight is treating detection as a set prediction problem. The model predicts a fixed-size set of $N$ predictions. During training, a bipartite matching is computed between predictions and ground-truth objects using the Hungarian algorithm:

$$\hat{\sigma} = \arg\min_{\sigma \in \mathfrak{S}_N} \sum_{i=1}^{N} \mathcal{L}_\text{match}(y_i, \hat{y}_{\sigma(i)})$$

The matching cost considers both class probability and bounding box similarity (L1 + GIoU):

$$\mathcal{L}_\text{match}(y_i, \hat{y}_{\sigma(i)}) = -\mathbb{1}_{\{c_i \neq \varnothing\}} \hat{p}_{\sigma(i)}(c_i) + \mathbb{1}_{\{c_i \neq \varnothing\}} \mathcal{L}_\text{box}(b_i, \hat{b}_{\sigma(i)})$$

After matching, the bipartite loss is:

$$\mathcal{L}_\text{Hungarian} = \sum_{i=1}^{N} \left[ -\log \hat{p}_{\hat{\sigma}(i)}(c_i) + \mathbb{1}_{\{c_i \neq \varnothing\}} \mathcal{L}_\text{box}(b_i, \hat{b}_{\hat{\sigma}(i)}) \right]$$

**Significance.** DETR eliminates NMS, anchors, and many heuristics, offering an elegant end-to-end trainable pipeline. Its limitations — slow convergence (500 epochs vs. 36 for Faster R-CNN) and poor performance on small objects — have been addressed by subsequent work including Deformable DETR (Zhu et al., 2021), which uses deformable attention to focus on a sparse set of key points, dramatically improving convergence and small-object detection.

---

## 22.3 Image Segmentation

While detection localizes objects with bounding boxes, segmentation provides pixel-level understanding.

### 22.3.1 Semantic Segmentation

Semantic segmentation assigns a class label to every pixel. There is no distinction between individual instances — all pixels of the same class share a label.

**FCN (Fully Convolutional Network).** Long et al. (2015) adapted classification networks into fully convolutional architectures that produce spatial output maps. By replacing fully connected layers with convolutional layers and using transposed convolutions for upsampling, FCN can accept arbitrary-size inputs and produce dense predictions. Skip connections from earlier layers combine coarse, semantic information with fine, spatial information.

**U-Net.** Ronneberger et al. (2015) introduced a symmetric encoder-decoder architecture with skip connections that concatenate encoder features with decoder features at corresponding resolutions. Originally designed for biomedical image segmentation with limited data, U-Net's architecture has become ubiquitous:

```
Encoder:               Decoder:
[Input: 572x572]       [Output: 388x388]
    ↓ conv+pool            ↑ upconv+concat
[284x284]              [..x..]
    ↓ conv+pool            ↑ upconv+concat
[140x140]              [..x..]
    ↓ conv+pool            ↑ upconv+concat
[68x68]                [..x..]
    ↓ conv+pool            ↑ upconv+concat
        [Bottleneck: 32x32]
```

Each encoder stage doubles the channels while halving spatial resolution; the decoder reverses this. Skip connections preserve fine-grained spatial information lost during downsampling.

**DeepLab.** The DeepLab family (Chen et al., 2017, 2018) introduced several key innovations:
- **Atrous (dilated) convolutions:** Increase receptive field without reducing resolution.
- **Atrous Spatial Pyramid Pooling (ASPP):** Multiple parallel atrous convolutions at different dilation rates capture multi-scale context.
- **DeepLabv3+** adds a simple decoder module and uses depthwise separable convolutions for efficiency.

### 22.3.2 Instance Segmentation

Instance segmentation combines detection and segmentation — it identifies each individual object instance and produces a pixel-level mask for each.

**Mask R-CNN (He et al., 2017).** Extends Faster R-CNN by adding a mask prediction branch parallel to the existing classification and box regression branches. For each detected object, a small FCN predicts a binary mask. Key innovation: *RoI Align* replaces RoI Pooling, using bilinear interpolation instead of quantization, preserving exact spatial alignment crucial for pixel-accurate masks.

The multi-task loss is:

$$\mathcal{L} = \mathcal{L}_\text{cls} + \mathcal{L}_\text{box} + \mathcal{L}_\text{mask}$$

where $\mathcal{L}_\text{mask}$ is a per-pixel binary cross-entropy loss, computed independently for each class (avoiding competition between classes in the mask branch).

### 22.3.3 Panoptic Segmentation

Panoptic segmentation (Kirillov et al., 2019) unifies semantic and instance segmentation: every pixel receives both a class label and an instance ID. "Things" (countable objects like cars, people) get instance IDs; "stuff" (amorphous regions like sky, grass) get only class labels. This provides a complete scene understanding, addressing the artificial divide between "things" and "stuff" in prior work.

### 22.3.4 SAM: Segment Anything Model

The Segment Anything Model (SAM) by Kirillov et al. (2023) represents a paradigm shift toward *promptable* and *foundation model* approaches to segmentation.

**Architecture.** SAM consists of three components:
1. **Image Encoder:** A ViT-H (huge) backbone, pretrained with MAE, encodes the image into embeddings. This runs once per image.
2. **Prompt Encoder:** Encodes sparse prompts (points, boxes, text) and dense prompts (masks) into embeddings.
3. **Mask Decoder:** A lightweight transformer decoder that combines image and prompt embeddings to predict segmentation masks.

**Promptable Segmentation.** SAM accepts diverse prompts:
- **Points:** Click on the object to segment (positive points) or background (negative points).
- **Bounding boxes:** Draw a box around the object.
- **Masks:** Provide a rough mask to refine.
- **Text:** Natural language descriptions (in SAM 2 and later variants).

**Training Data.** SAM was trained on SA-1B, a dataset of 11 million images with over 1.1 billion masks, collected through a data engine that iteratively used model predictions to accelerate annotation. This massive scale enables zero-shot generalization to new domains.

```python
from segment_anything import sam_model_registry, SamPredictor

# Load SAM model
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
predictor = SamPredictor(sam)

# Set image (run encoder once)
predictor.set_image(image)  # image: (H, W, 3) numpy array

# Segment with point prompt
input_point = np.array([[500, 375]])   # (x, y) coordinate
input_label = np.array([1])            # 1 = foreground, 0 = background

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True  # Returns 3 masks at different granularities
)

# Select highest-scoring mask
best_mask = masks[scores.argmax()]  # (H, W) boolean array

# Segment with box prompt
input_box = np.array([100, 100, 400, 400])  # [x1, y1, x2, y2]
masks, scores, logits = predictor.predict(
    box=input_box,
    multimask_output=False
)
```

---

## 22.4 3D Vision

Understanding three-dimensional structure from 2D images has always been a central goal of computer vision. Recent advances in neural rendering and point cloud processing have dramatically advanced the field.

### 22.4.1 NeRF: Neural Radiance Fields

Neural Radiance Fields (Mildenhall et al., 2020) represent a 3D scene as a continuous volumetric function, enabling photorealistic novel view synthesis from a sparse set of input photographs.

**Core Representation.** A NeRF models a scene as a continuous function $F_\Theta$ that maps a 3D position $\mathbf{x} = (x, y, z)$ and viewing direction $\mathbf{d} = (\theta, \phi)$ to a color $\mathbf{c} = (r, g, b)$ and volume density $\sigma$:

$$F_\Theta : (\mathbf{x}, \mathbf{d}) \rightarrow (\mathbf{c}, \sigma)$$

This function is parameterized by an MLP. The density $\sigma$ depends only on position (it is a property of the scene geometry), while color depends on both position and viewing direction (to model view-dependent effects like specularities).

**Volume Rendering.** To render a pixel, a ray $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$ is cast from the camera origin $\mathbf{o}$ through the pixel in direction $\mathbf{d}$. The expected color along the ray is computed via the volume rendering equation:

$$C(\mathbf{r}) = \int_{t_n}^{t_f} T(t) \, \sigma(\mathbf{r}(t)) \, \mathbf{c}(\mathbf{r}(t), \mathbf{d}) \, dt$$

where $T(t) = \exp\left(-\int_{t_n}^{t} \sigma(\mathbf{r}(s)) \, ds\right)$ is the accumulated transmittance — the probability that the ray travels from $t_n$ to $t$ without hitting anything.

In practice, this integral is approximated by stratified sampling along the ray and quadrature:

$$\hat{C}(\mathbf{r}) = \sum_{i=1}^{N} T_i \, (1 - \exp(-\sigma_i \delta_i)) \, \mathbf{c}_i, \quad T_i = \exp\left(-\sum_{j=1}^{i-1} \sigma_j \delta_j\right)$$

where $\delta_i = t_{i+1} - t_i$ is the distance between adjacent samples.

**Positional Encoding.** MLPs struggle to learn high-frequency content directly from low-dimensional inputs. NeRF applies a positional encoding $\gamma$ that maps coordinates to a higher-dimensional space using sinusoidal functions:

$$\gamma(p) = \left(\sin(2^0 \pi p), \cos(2^0 \pi p), \ldots, \sin(2^{L-1} \pi p), \cos(2^{L-1} \pi p)\right)$$

This enables the MLP to represent fine geometric and appearance details. Typically, $L = 10$ for spatial coordinates and $L = 4$ for viewing directions.

**MLP Architecture.** The NeRF MLP consists of 8 fully connected layers with ReLU activations and 256 channels. The positional-encoded position is input at the first layer and also concatenated at the 5th layer (skip connection). After 8 layers, density $\sigma$ is output, and the viewing direction is concatenated before a final layer predicts RGB color.

**Training.** NeRF is trained by minimizing the photometric loss between rendered and ground-truth pixel colors:

$$\mathcal{L} = \sum_{\mathbf{r} \in \mathcal{R}} \| \hat{C}(\mathbf{r}) - C(\mathbf{r}) \|_2^2$$

A hierarchical sampling strategy with "coarse" and "fine" networks improves efficiency — the coarse network identifies important regions along each ray, and the fine network concentrates samples there.

**Limitations and Extensions.** NeRF's main limitations are slow training (hours per scene) and slow rendering (~30 seconds per frame). Subsequent work has addressed these: Instant-NGP (Mueller et al., 2022) uses hash-based positional encoding for near-instant training; Mip-NeRF (Barron et al., 2021) handles multi-scale rendering; and various methods enable dynamic scenes, relighting, and large-scale outdoor reconstruction.

### 22.4.2 3D Gaussian Splatting

3D Gaussian Splatting (Kerbl et al., 2023) offers a radically different approach to novel view synthesis — using an explicit, point-based representation instead of an implicit neural field.

**Representation.** A scene is represented as a set of 3D Gaussians, each parameterized by:
- **Position** $\boldsymbol{\mu} \in \mathbb{R}^3$: center of the Gaussian
- **Covariance** $\boldsymbol{\Sigma} \in \mathbb{R}^{3 \times 3}$: shape and orientation (stored as a rotation quaternion $\mathbf{q}$ and scale vector $\mathbf{s}$ for guaranteed positive semi-definiteness: $\boldsymbol{\Sigma} = \mathbf{R} \mathbf{S} \mathbf{S}^T \mathbf{R}^T$)
- **Opacity** $\alpha \in [0, 1]$
- **Color** represented via spherical harmonics coefficients for view-dependent appearance

Each Gaussian has the form:

$$G(\mathbf{x}) = \exp\left(-\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)$$

**Differentiable Rasterization.** Instead of ray marching (as in NeRF), 3D Gaussian Splatting uses differentiable rasterization:
1. Project 3D Gaussians to 2D screen space using the camera projection.
2. Sort Gaussians by depth per tile.
3. For each pixel, alpha-composite the overlapping 2D Gaussians front-to-back:

$$C = \sum_{i=1}^{N} \mathbf{c}_i \alpha_i \prod_{j=1}^{i-1} (1 - \alpha_j)$$

This tile-based rasterization is massively parallelizable on GPUs.

**Adaptive Density Control.** During training, the system adaptively adds and removes Gaussians:
- **Densification:** Gaussians with large gradients (indicating under-reconstruction) are split or cloned.
- **Pruning:** Nearly transparent Gaussians ($\alpha$ near 0) are removed.

**Advantages over NeRF.** 3D Gaussian Splatting achieves:
- **Real-time rendering** at 30+ fps (vs. minutes for NeRF)
- **Faster training** (minutes vs. hours)
- **Comparable or superior visual quality**
- **Explicit representation** that is easier to edit and manipulate

### 22.4.3 Point Cloud Processing

Point clouds — unordered sets of 3D points — arise from LiDAR, depth cameras, and structure-from-motion. Processing them requires architectures that handle unordered sets and varying cardinalities.

**PointNet (Qi et al., 2017).** The foundational architecture for point cloud processing directly consumes raw point clouds. Key design principles:
- **Permutation invariance:** Uses a symmetric function (max pooling) to aggregate point-wise features into a global feature vector, ensuring the output is invariant to input ordering.
- **Per-point MLPs:** Shared MLPs process each point independently before aggregation.
- **Spatial Transformer Networks (T-Nets):** Small networks that predict affine transformations to align the point cloud, providing invariance to geometric transformations.

$$f(\{x_1, \ldots, x_n\}) = g(\text{MAX}_{i=1,\ldots,n} \{h(x_i)\})$$

where $h$ is a per-point MLP and $g$ is a classification/segmentation head.

**PointNet++ (Qi et al., 2017).** Extends PointNet with hierarchical feature learning. It recursively applies PointNet to local neighborhoods at multiple scales:
1. **Sampling:** Farthest point sampling selects representative points.
2. **Grouping:** Ball query finds neighbors within a radius.
3. **PointNet:** Applied to each local group to extract local features.
4. **Hierarchical abstraction:** Repeated at multiple levels for increasingly global context.

---

## 22.5 Video Understanding

Video introduces the temporal dimension, requiring models that capture both spatial appearance and temporal dynamics.

### 22.5.1 Temporal Modeling with 3D Convolutions

**3D Convolutions.** The straightforward extension of 2D convolutions to video applies convolutional kernels across both spatial and temporal dimensions. A 3D convolutional kernel of size $k_t \times k_h \times k_w$ slides across the temporal, height, and width dimensions of a video tensor.

**I3D (Inflated 3D ConvNets).** Carreira and Zisserman (2017) proposed inflating pretrained 2D convolutional kernels into 3D by repeating the 2D weights along the temporal dimension and dividing by $k_t$ to preserve the activation magnitude. This "inflation" strategy leverages strong ImageNet pretraining while enabling temporal modeling. I3D processes both RGB frames and optical flow streams, fusing their predictions.

**Factorized Convolutions.** Full 3D convolutions are computationally expensive. Factorization approaches decompose them:
- **R(2+1)D:** Decomposes 3D convolution into a 2D spatial convolution followed by a 1D temporal convolution, adding nonlinearity between them and reducing parameters.
- **SlowFast Networks (Feichtenhofer et al., 2019):** Two-pathway architecture where a "Slow" pathway operates at low frame rate for spatial semantics and a "Fast" pathway operates at high frame rate for temporal dynamics.

### 22.5.2 Video Transformers

**TimeSFormer (Bertasius et al., 2021).** Extends ViT to video by adding temporal attention. Rather than computing joint space-time attention (which is prohibitively expensive), TimeSFormer uses *divided space-time attention*: temporal attention across frames at the same spatial location, followed by spatial attention within each frame.

**ViViT (Arnab et al., 2021).** Explores multiple strategies for adapting ViT to video:
1. **Spatio-temporal attention:** Full attention over all space-time tokens (most accurate, most expensive).
2. **Factorized encoder:** Separate spatial and temporal transformers.
3. **Factorized self-attention:** Spatial and temporal attention within the same encoder.
4. **Factorized dot-product attention:** Decompose attention heads into spatial and temporal groups.

**VideoLLaMA.** Recent video-language models extend LLMs to understand video by encoding video frames (or clips) as visual tokens and integrating them into the language model's context. VideoLLaMA and similar models enable video question-answering, captioning, and temporal grounding through multimodal instruction tuning.

### 22.5.3 Action Recognition

Action recognition classifies activities in video clips. The field has evolved from hand-crafted features (improved Dense Trajectories) through 3D CNNs (C3D, I3D) to Transformer-based approaches. Modern models operate on short clips (typically 8-32 frames) and use temporal pooling or classification tokens for video-level predictions. Datasets include Kinetics-400/600/700, Something-Something V2 (which requires temporal reasoning), and ActivityNet for long-form videos.

---

## 22.6 Pose Estimation

Human pose estimation recovers the 2D or 3D positions of body joints (keypoints) from images or video. It enables applications in fitness tracking, sign language recognition, motion capture, and human-computer interaction.

### 22.6.1 Top-Down vs Bottom-Up Approaches

**Top-down** methods first detect each person with a bounding box, then estimate keypoints within each box independently. This approach excels at accuracy but scales linearly with the number of people.

**Bottom-up** methods detect all keypoints in the image simultaneously, then group them into individual people. This is faster for multi-person scenes (runtime independent of person count) but requires sophisticated grouping algorithms.

### 22.6.2 Key Systems

**OpenPose (Cao et al., 2019).** A pioneering bottom-up multi-person pose estimation system. It predicts two types of maps:
1. **Confidence maps** for body part locations (one map per keypoint type).
2. **Part Affinity Fields (PAFs)** — 2D vector fields encoding the direction between connected parts, used to associate keypoints belonging to the same person.

A greedy matching algorithm uses PAFs to assemble detected keypoints into full-body poses.

**MediaPipe Pose.** Google's MediaPipe provides lightweight, real-time pose estimation optimized for mobile and edge devices. It uses a two-stage pipeline: a person detector (BlazePose detector) followed by a pose landmark model that estimates 33 keypoints per person, including body, hand, and face landmarks. The models use depthwise separable convolutions and are quantized for on-device inference.

**ViTPose (Xu et al., 2022).** Demonstrates that a plain Vision Transformer backbone, without task-specific architectural modifications, can achieve state-of-the-art results on human pose estimation when properly pretrained (e.g., with MAE). ViTPose uses a simple decoder head on top of a ViT backbone and achieves competitive results across body, hand, face, and whole-body pose estimation. This reinforces the trend of general-purpose vision backbones replacing task-specific architectures.

```python
import mediapipe as mp
import cv2

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:
    cap = cv2.VideoCapture(0)  # Webcam

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

            # Access individual landmarks
            landmarks = results.pose_landmarks.landmark
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            print(f"Left shoulder: ({left_shoulder.x:.2f}, "
                  f"{left_shoulder.y:.2f}, {left_shoulder.z:.2f})")

        cv2.imshow('Pose Estimation', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
```

---

## 22.7 Optical Character Recognition (OCR)

OCR extracts text from images — a problem that spans from reading license plates to digitizing centuries-old manuscripts. Modern OCR systems have evolved from traditional pipelines to end-to-end deep learning approaches.

### 22.7.1 Traditional Pipeline: Detection + Recognition

The classic OCR pipeline consists of two stages:

1. **Text Detection:** Localize text regions in the image. Methods include:
   - **EAST (Efficient and Accurate Scene Text):** Predicts text region geometry (rotated boxes or quadrilaterals) directly from feature maps.
   - **DBNet (Differentiable Binarization):** Uses a differentiable binarization module to produce sharp text region boundaries, replacing the non-differentiable threshold step.
   - **CRAFT (Character Region Awareness for Text detection):** Detects individual characters and their affinities to group them into words.

2. **Text Recognition:** Convert each detected text region into a character sequence. Approaches include:
   - **CRNN (Convolutional Recurrent Neural Network):** CNN feature extraction followed by a bidirectional LSTM and CTC (Connectionist Temporal Classification) decoding.
   - **Attention-based:** Encoder-decoder with attention, treating recognition as sequence-to-sequence translation.

### 22.7.2 End-to-End Approaches

**TrOCR (Li et al., 2023).** A Transformer-based end-to-end OCR model that uses a pretrained image Transformer (ViT or DeiT) as the encoder and a pretrained language model (RoBERTa or GPT-2) as the decoder. TrOCR eliminates the need for external text detection by operating on cropped text line images. The key insight is leveraging both visual and linguistic pretraining:

```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

# Load pretrained TrOCR
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed")

# Recognize text from a cropped text image
image = Image.open("text_line.png").convert("RGB")
pixel_values = processor(images=image, return_tensors="pt").pixel_values

generated_ids = model.generate(pixel_values, max_new_tokens=64)
text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(f"Recognized text: {text}")
```

**PaddleOCR.** An open-source OCR toolkit from Baidu that provides a complete pipeline:
- **PP-OCR:** Lightweight pipeline (detection + direction classifier + recognition) optimized for deployment. The v4 model achieves strong accuracy with a total model size of ~15MB.
- **PP-Structure:** Document structure analysis including table recognition and layout analysis.
- Supports 80+ languages with multilingual models.

### 22.7.3 Document AI at Scale

Modern document understanding goes beyond character-level OCR to comprehend document structure:
- **LayoutLM / LayoutLMv3 (Huang et al., 2022):** Multimodal pretrained models that jointly learn text, layout (bounding box positions), and visual features. They can perform document classification, key information extraction, and table understanding.
- **Donut (Kim et al., 2022):** An OCR-free document understanding model that uses a Swin Transformer encoder and GPT decoder to directly generate structured output from document images, eliminating the OCR preprocessing step entirely.

---

## 22.8 Depth Estimation

Depth estimation recovers 3D structure from images, enabling applications in autonomous driving, robotics, AR/VR, and 3D reconstruction.

### 22.8.1 Monocular Depth Estimation

Estimating depth from a single image is inherently ill-posed — infinitely many 3D scenes can project to the same 2D image. Deep learning approaches learn strong priors from large-scale data.

**MiDaS (Ranftl et al., 2020).** A monocular depth estimation model trained on a mix of diverse datasets with *affine-invariant* loss functions. The key insight: different depth datasets use different depth representations (metric depth, disparity, ordinal depth), making joint training challenging. MiDaS introduces losses that are invariant to unknown scale and shift, enabling mixing of diverse training data:

$$\mathcal{L} = \frac{1}{2M} \sum_{i=1}^{M} \rho(d_i^* - (\hat{s} d_i + \hat{t}))$$

where $\hat{s}$ and $\hat{t}$ are the least-squares scale and shift aligning predictions to ground truth.

**DPT (Dense Prediction Transformer).** Ranftl et al. (2021) replaced the CNN backbone with a ViT, demonstrating that transformer features produce significantly more coherent depth maps with better global structure. DPT reassembles multi-scale features from ViT tokens using a convolutional decoder, achieving state-of-the-art monocular depth estimation.

**Depth Anything (Yang et al., 2024).** A foundation model for monocular depth estimation trained on a massive dataset of 62 million images (labeled and unlabeled) using a combination of supervised learning and self-supervised consistency regularization. Depth Anything achieves remarkable zero-shot generalization and has become a popular building block for downstream 3D tasks.

```python
import torch
from transformers import DPTForDepthEstimation, DPTImageProcessor
from PIL import Image
import numpy as np

# Load DPT model
processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

# Estimate depth
image = Image.open("scene.jpg")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    depth = outputs.predicted_depth

# Interpolate to original image size
depth = torch.nn.functional.interpolate(
    depth.unsqueeze(1),
    size=image.size[::-1],  # (height, width)
    mode="bicubic",
    align_corners=False
).squeeze()

# Normalize for visualization
depth_np = depth.numpy()
depth_normalized = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())
depth_image = (depth_normalized * 255).astype(np.uint8)
```

### 22.8.2 Stereo Matching

Stereo depth estimation uses two calibrated cameras to compute depth via triangulation. The core problem is *stereo matching* — finding corresponding points between left and right images.

The depth at a pixel is inversely proportional to the disparity $d$:

$$Z = \frac{f \cdot B}{d}$$

where $f$ is the focal length and $B$ is the baseline (distance between cameras).

Modern stereo methods like RAFT-Stereo (Lipson et al., 2021) and CREStereo use iterative refinement of disparity maps, analogous to optical flow estimation. They construct 4D correlation volumes from feature maps and iteratively update disparity estimates using GRU-based modules.

### 22.8.3 Depth from Multi-View

Multi-view stereo (MVS) reconstructs 3D geometry from multiple overlapping images. Learning-based approaches like MVSNet (Yao et al., 2018) construct plane-sweep cost volumes from multiple views and use 3D CNNs to regularize them. The pipeline typically involves:

1. **Feature extraction:** 2D CNN extracts features from each view.
2. **Cost volume construction:** Features are warped to reference camera frustum at multiple depth hypotheses.
3. **Cost volume regularization:** 3D CNN or recurrent processing smooths and refines the volume.
4. **Depth regression:** Soft argmin (differentiable) extracts depth from the regularized volume.

Multi-view approaches produce metric depth (unlike monocular methods which produce relative depth) and serve as the foundation for large-scale 3D reconstruction in applications like autonomous driving, mapping, and digital twins.

---

## 22.9 Putting It All Together: The Modern Vision Stack

Modern computer vision systems rarely use a single technique in isolation. A comprehensive visual understanding pipeline might combine:

1. **Backbone:** Swin Transformer or ViT (with MAE pretraining) for feature extraction.
2. **Detection:** DETR or YOLOv8 for object localization.
3. **Segmentation:** SAM for promptable, zero-shot segmentation.
4. **Depth:** DPT or Depth Anything for monocular depth.
5. **3D Reconstruction:** 3D Gaussian Splatting for novel view synthesis.
6. **Pose:** ViTPose for human pose estimation.
7. **OCR:** LayoutLMv3 for document understanding.
8. **Video:** VideoLLM for temporal reasoning and video QA.

The trend is unmistakable: transformer-based architectures are unifying computer vision under a common framework, and foundation models pretrained on massive data are replacing task-specific pipelines. The vision engineer's role is shifting from designing architectures to composing and fine-tuning powerful pretrained models.

---

## Exercises

### Conceptual Questions

1. **ViT vs CNN inductive biases.** Explain why ViT requires more data than CNNs to achieve comparable performance. What specific inductive biases do CNNs have that ViT lacks, and how does data compensate for this?

2. **DETR's Hungarian matching.** Why is bipartite matching necessary in DETR? What would happen if each prediction slot were independently assigned to the nearest ground-truth object?

3. **NeRF vs 3D Gaussian Splatting.** Compare the implicit representation of NeRF with the explicit representation of 3D Gaussian Splatting. What are the fundamental tradeoffs in terms of rendering speed, editability, and memory?

4. **SAM's zero-shot capability.** Explain how SAM achieves zero-shot segmentation on unseen domains. What role does the scale of training data (SA-1B) play versus architectural design?

### Implementation Exercises

5. **Implement a minimal ViT.** Using only PyTorch, implement a Vision Transformer from scratch. Train it on CIFAR-10. Compare its performance with and without data augmentation. Experiment with different patch sizes (4, 8, 16) and analyze the speed-accuracy tradeoff.

6. **Object detection with YOLOv8.** Fine-tune YOLOv8 on a custom dataset (e.g., a specific domain like medical images, satellite imagery, or retail products). Evaluate using mAP@0.5 and mAP@0.5:0.95. Analyze failure cases.

7. **SAM integration.** Build a pipeline that combines YOLOv8 detection with SAM segmentation: use YOLOv8 to detect objects, then use each detection's bounding box as a prompt to SAM for pixel-level masks. Compare with Mask R-CNN on the same dataset.

8. **Monocular depth pipeline.** Use DPT to estimate depth from a video. Implement temporal consistency filtering (e.g., exponential moving average) to reduce flickering between frames. Visualize the depth maps as a colored point cloud.

### Research Questions

9. **Vision foundation models.** Compare three recent vision foundation models (DINOv2, SAM, Florence-2). How do they differ in pretraining strategy, architecture, and downstream task performance? What does this suggest about the future of task-specific models?

10. **Video understanding challenge.** Why is temporal reasoning in video still significantly harder than spatial understanding in images? Design an experiment to quantify this gap using Something-Something V2 (which requires temporal reasoning) versus Kinetics (which can often be solved from individual frames).

---

## References

1. Arnab, A., Dehghani, M., Heigold, G., Sun, C., Lucic, M., & Schmidhuber, J. (2021). ViViT: A Video Vision Transformer. *ICCV*.

2. Barron, J. T., Mildenhall, B., Tancik, M., Hedman, P., Martin-Brualla, R., & Srinivasan, P. P. (2021). Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields. *ICCV*.

3. Bertasius, G., Wang, H., & Torresani, L. (2021). Is Space-Time Attention All You Need for Video Understanding? *ICML*.

4. Cao, Z., Hidalgo, G., Simon, T., Wei, S.-E., & Sheikh, Y. (2019). OpenPose: Realtime Multi-Person 2D Pose Estimation Using Part Affinity Fields. *IEEE TPAMI*.

5. Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., & Zagoruyko, S. (2020). End-to-End Object Detection with Transformers. *ECCV*.

6. Carreira, J., & Zisserman, A. (2017). Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset. *CVPR*.

7. Chen, L.-C., Papandreou, G., Kokkinos, I., Murphy, K., & Yuille, A. L. (2018). DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs. *IEEE TPAMI*.

8. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR*.

9. Feichtenhofer, C., Fan, H., Malik, J., & He, K. (2019). SlowFast Networks for Video Recognition. *ICCV*.

10. Girshick, R. (2015). Fast R-CNN. *ICCV*.

11. Girshick, R., Donahue, J., Darrell, T., & Malik, J. (2014). Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation. *CVPR*.

12. He, K., Gkioxari, G., Dollar, P., & Girshick, R. (2017). Mask R-CNN. *ICCV*.

13. Huang, Y., Lv, T., Cui, L., Lu, Y., & Wei, F. (2022). LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking. *ACM MM*.

14. Jocher, G. (2023). Ultralytics YOLOv8. https://github.com/ultralytics/ultralytics.

15. Kerbl, B., Kopanas, G., Leimkuhler, T., & Drettakis, G. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering. *ACM TOG (SIGGRAPH)*.

16. Kim, G., Hong, T., Yim, M., Nam, J., Park, J., Yim, J., ... & Park, S. (2022). OCR-Free Document Understanding Transformer. *ECCV*.

17. Kirillov, A., He, K., Girshick, R., Rother, C., & Dollar, P. (2019). Panoptic Segmentation. *CVPR*.

18. Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., ... & Girshick, R. (2023). Segment Anything. *ICCV*.

19. Li, M., Lv, T., Chen, J., Cui, L., Lu, Y., Florencio, D., ... & Wei, F. (2023). TrOCR: Transformer-Based Optical Character Recognition with Pre-trained Models. *AAAI*.

20. Lipson, L., Teed, Z., & Deng, J. (2021). RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching. *3DV*.

21. Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., ... & Guo, B. (2021). Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows. *ICCV*.

22. Long, J., Shelhamer, E., & Darrell, T. (2015). Fully Convolutional Networks for Semantic Segmentation. *CVPR*.

23. Mildenhall, B., Srinivasan, P. P., Tancik, M., Barron, J. T., Ramamoorthi, R., & Ng, R. (2020). NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. *ECCV*.

24. Mueller, T., Evans, A., Schied, C., & Keller, A. (2022). Instant Neural Graphics Primitives with a Multiresolution Hash Encoding. *ACM TOG (SIGGRAPH)*.

25. Qi, C. R., Su, H., Mo, K., & Guibas, L. J. (2017). PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation. *CVPR*.

26. Qi, C. R., Yi, L., Su, H., & Guibas, L. J. (2017). PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space. *NeurIPS*.

27. Ranftl, R., Bochkovskiy, A., & Koltun, V. (2021). Vision Transformers for Dense Prediction. *ICCV*.

28. Ranftl, R., Lasinger, K., Hafner, D., Schindler, K., & Koltun, V. (2020). Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot Cross-Dataset Transfer. *IEEE TPAMI*.

29. Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You Only Look Once: Unified, Real-Time Object Detection. *CVPR*.

30. Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. *NeurIPS*.

31. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI*.

32. Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A., & Jegou, H. (2021). Training Data-Efficient Image Transformers & Distillation Through Attention. *ICML*.

33. Xu, Y., Zhang, J., Zhang, Q., & Tao, D. (2022). ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation. *NeurIPS*.

34. Yang, L., Kang, B., Huang, Z., Xu, X., Feng, J., & Zhao, H. (2024). Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data. *CVPR*.

35. Yao, Y., Luo, Z., Li, S., Fang, T., & Quan, L. (2018). MVSNet: Depth Inference for Unstructured Multi-View Stereo. *ECCV*.

36. Zhu, X., Su, W., Lu, L., Li, B., Wang, X., & Dai, J. (2021). Deformable DETR: Deformable Transformers for End-to-End Object Detection. *ICLR*.
