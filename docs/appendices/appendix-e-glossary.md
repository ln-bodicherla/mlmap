# Appendix E: Glossary of Terms

This glossary defines the key terms used throughout the book. Each entry includes the term, a concise definition, and a reference to the chapter where the term is first introduced or most thoroughly discussed. Terms are organized alphabetically.

---

**A/B Testing.** A controlled experiment comparing two variants (A and B) to determine which performs better on a specified metric. Used in production ML systems to evaluate model changes against a baseline. *Ch 19*

**Ablation Study.** An experimental technique where components of a model or system are systematically removed or disabled to measure their individual contributions to overall performance. *Ch 7*

**Activation Function.** A non-linear function applied to the output of a neuron in a neural network. Common examples include ReLU, sigmoid, tanh, GELU, and SiLU/Swish. Without activation functions, a multi-layer network collapses to a single linear transformation. *Ch 7*

**Adam (Adaptive Moment Estimation).** An optimization algorithm that maintains per-parameter adaptive learning rates using estimates of the first and second moments of the gradients. Widely used as the default optimizer in deep learning. *Ch 3*

**AdamW.** A variant of Adam that decouples weight decay from the gradient update, correcting a subtle bug in the original Adam formulation. Standard in modern LLM training. *Ch 3*

**Adversarial Example.** An input deliberately crafted to cause a machine learning model to make a mistake, typically by adding imperceptible perturbations to a legitimate input. *Ch 26*

**Agent.** In the context of AI, a system that uses a language model to reason, plan, and take actions in an environment, often by calling external tools or APIs. *Ch 16*

**Alignment.** The process of training AI models to behave in accordance with human values, intentions, and preferences. Techniques include RLHF, DPO, and constitutional AI. *Ch 9*

**Attention Mechanism.** A neural network component that computes a weighted combination of values based on the similarity between queries and keys. Enables models to focus on relevant parts of the input. *Ch 7*

**Autograd.** An automatic differentiation system that computes gradients of tensors with respect to their inputs by recording operations in a computational graph. The core of PyTorch's training capability. *Ch 6*

**Autoregressive Model.** A model that generates output one token at a time, where each token is conditioned on all previously generated tokens. GPT-family models are autoregressive. *Ch 8*

**Backpropagation.** An algorithm for computing gradients of a loss function with respect to all parameters in a neural network by applying the chain rule of calculus recursively from output to input. *Ch 3, 7*

**Batch Normalization.** A technique that normalizes the activations of each layer across a mini-batch during training, stabilizing and accelerating convergence. *Ch 7*

**Batch Size.** The number of training examples processed together in one forward and backward pass. Larger batch sizes increase throughput but require more memory and may affect generalization. *Ch 6*

**Bayes' Theorem.** A formula for updating the probability of a hypothesis given new evidence: P(A|B) = P(B|A) * P(A) / P(B). Foundational to probabilistic machine learning. *Ch 4*

**Beam Search.** A heuristic search algorithm that explores multiple candidate sequences simultaneously during decoding, keeping only the top-k most promising at each step. *Ch 18*

**Bias (Statistical).** The systematic error introduced by approximating a complex real-world problem with a simplified model. High bias leads to underfitting. *Ch 5*

**Bias-Variance Tradeoff.** The tension between a model's ability to fit training data (low bias) and its ability to generalize to unseen data (low variance). A fundamental concept in machine learning. *Ch 5*

**BPE (Byte-Pair Encoding).** A subword tokenization algorithm that iteratively merges the most frequent pair of characters or tokens in a corpus. Used by GPT models and many others. *Ch 10*

**Chain-of-Thought (CoT) Prompting.** A prompting technique that encourages a language model to show its reasoning step by step before arriving at a final answer, improving performance on reasoning tasks. *Ch 16*

**Checkpoint.** A saved snapshot of a model's parameters (and optionally optimizer state) at a specific point during training, enabling recovery from failures and model selection. *Ch 6*

**Chinchilla Scaling Laws.** Empirical laws established by Hoffmann et al. (2022) showing that for a given compute budget, model size and training data should be scaled equally, and that most existing LLMs were significantly undertrained. *Ch 11*

**Classification.** A supervised learning task where the model predicts a discrete label (class) for each input. Binary classification has two classes; multiclass has more. *Ch 5*

**CLIP (Contrastive Language-Image Pre-training).** A model trained by OpenAI to learn visual concepts from natural language supervision, enabling zero-shot image classification and vision-language tasks. *Ch 17*

**Clustering.** An unsupervised learning task that groups data points into clusters based on similarity without predefined labels. Common algorithms include k-means, DBSCAN, and hierarchical clustering. *Ch 5*

**CNN (Convolutional Neural Network).** A neural network architecture that uses convolutional layers to automatically learn spatial hierarchies of features. Widely used in computer vision. *Ch 7, 22*

**Compute-Optimal Training.** Training a model with the optimal allocation of compute budget between model size and number of training tokens, as defined by scaling laws. *Ch 11*

**Constitutional AI.** An alignment approach where the AI is trained to follow a set of principles (a "constitution") rather than relying solely on human feedback for every decision. *Ch 9*

**Contrastive Learning.** A self-supervised learning paradigm where the model learns representations by pulling similar examples together and pushing dissimilar examples apart in embedding space. *Ch 17*

**Cross-Entropy Loss.** A loss function commonly used for classification tasks that measures the divergence between the predicted probability distribution and the true distribution. *Ch 5, 7*

**Cross-Validation.** A model evaluation technique that partitions data into multiple folds, training on some folds and validating on the remaining fold, rotating until every fold has served as validation. k-fold cross-validation is the most common variant. *Ch 5*

**CUDA (Compute Unified Device Architecture).** NVIDIA's parallel computing platform and API that enables GPU-accelerated computation. Essential infrastructure for deep learning training. *Ch 6*

**Data Augmentation.** Techniques for artificially expanding a training dataset by applying transformations (rotations, crops, noise, etc.) to existing examples. Reduces overfitting and improves generalization. *Ch 7, 22*

**Data Parallelism.** A distributed training strategy where the training data is split across multiple devices, each holding a complete copy of the model. Gradients are synchronized across devices after each step. *Ch 12*

**DataLoader.** A PyTorch utility that provides an iterable over a dataset with support for batching, shuffling, multi-process data loading, and custom sampling. *Ch 6*

**DDP (DistributedDataParallel).** PyTorch's primary API for distributed data-parallel training across multiple GPUs or nodes. Uses all-reduce to synchronize gradients. *Ch 12*

**Decision Tree.** A supervised learning model that makes predictions by learning a hierarchy of if-then-else rules from the features. The basis for random forests and gradient boosting. *Ch 5*

**Deep Learning.** A subfield of machine learning that uses neural networks with many layers (hence "deep") to learn hierarchical representations from data. *Ch 7*

**Differential Privacy.** A mathematical framework providing formal guarantees that the output of a computation does not reveal whether any individual's data was included in the input. Applied to ML via DP-SGD. *Ch 26*

**Diffusion Model.** A generative model that learns to reverse a gradual noising process, generating data by iteratively denoising from random noise. Powers state-of-the-art image generation systems. *Ch 17*

**Dimensionality Reduction.** Techniques for reducing the number of features (dimensions) in a dataset while preserving important structure. Common methods include PCA, t-SNE, and UMAP. *Ch 5*

**DPO (Direct Preference Optimization).** An alignment technique that fine-tunes a language model directly on preference data without training a separate reward model, simplifying the RLHF pipeline. *Ch 9*

**Dropout.** A regularization technique where randomly selected neurons are ignored (set to zero) during each training step, preventing co-adaptation and reducing overfitting. *Ch 7*

**Eigenvalue / Eigenvector.** For a square matrix A, an eigenvector v satisfies Av = lambda * v, where lambda is the corresponding eigenvalue. Fundamental to PCA, spectral methods, and understanding linear transformations. *Ch 2*

**Embedding.** A dense, low-dimensional vector representation of a discrete object (word, token, user, item). Embeddings capture semantic similarity: similar objects have nearby embeddings. *Ch 7, 8*

**Epoch.** One complete pass through the entire training dataset during model training. Models are typically trained for multiple epochs. *Ch 6*

**Evaluation Metric.** A quantitative measure of model performance. Common metrics include accuracy, precision, recall, F1 score, AUC-ROC, mean squared error, and perplexity. *Ch 5*

**Feature Engineering.** The process of creating, selecting, and transforming input features to improve model performance. Traditionally a manual process; deep learning automates much of this. *Ch 5*

**Federated Learning.** A distributed ML approach where models are trained across multiple decentralized devices holding local data, without exchanging the raw data. Preserves privacy. *Ch 26*

**Few-Shot Learning.** A learning paradigm where a model performs a task given only a few examples, without additional training. LLMs demonstrate few-shot learning through in-context examples in the prompt. *Ch 8*

**Fine-Tuning.** The process of taking a pretrained model and further training it on a smaller, task-specific dataset to adapt it for a particular use case. *Ch 8*

**FlashAttention.** An IO-aware exact attention algorithm that reduces memory usage from quadratic to linear and significantly speeds up Transformer training and inference. *Ch 14*

**FLOPs (Floating-Point Operations).** A measure of computational cost, counting the number of floating-point arithmetic operations required. Used to estimate and compare training costs. *Ch 11*

**FSDP (Fully Sharded Data Parallel).** A PyTorch distributed training strategy that shards model parameters, gradients, and optimizer states across devices, reducing per-device memory. Based on the ZeRO algorithm. *Ch 12*

**GAN (Generative Adversarial Network).** A generative model consisting of two networks -- a generator and a discriminator -- trained adversarially. The generator creates synthetic data while the discriminator tries to distinguish real from fake. *Ch 17*

**GELU (Gaussian Error Linear Unit).** An activation function defined as x * Phi(x), where Phi is the Gaussian CDF. Standard in Transformer architectures. *Ch 7*

**Gradient.** A vector of partial derivatives indicating the direction and magnitude of the steepest increase of a function. In ML, gradients of the loss with respect to parameters guide optimization. *Ch 3*

**Gradient Accumulation.** A technique that simulates larger batch sizes by accumulating gradients over multiple forward-backward passes before performing a parameter update. Useful when GPU memory is limited. *Ch 12*

**Gradient Clipping.** A technique that limits the magnitude of gradients during training to prevent exploding gradients. Common strategies include clipping by norm and clipping by value. *Ch 7*

**Gradient Descent.** An iterative optimization algorithm that updates parameters in the direction opposite to the gradient of the loss function, moving toward a (local) minimum. *Ch 3*

**Graph Neural Network (GNN).** A neural network architecture designed to operate on graph-structured data, learning node, edge, or graph-level representations through message passing between connected nodes. *Ch 24*

**Greedy Decoding.** A text generation strategy that selects the highest-probability token at each step. Simple but can lead to suboptimal outputs compared to beam search or sampling. *Ch 18*

**Grouped-Query Attention (GQA).** An attention variant that uses fewer key-value heads than query heads, interpolating between multi-head attention and multi-query attention. Reduces memory during inference while maintaining quality. *Ch 7*

**Hallucination.** In the context of LLMs, the generation of text that is fluent and plausible-sounding but factually incorrect or fabricated. A major challenge for deployed LLM systems. *Ch 15*

**Hyperparameter.** A parameter that controls the learning process itself (e.g., learning rate, batch size, number of layers) rather than being learned from data. Typically set before training begins. *Ch 5*

**In-Context Learning.** The ability of large language models to perform tasks based on examples or instructions provided in the prompt, without updating model weights. *Ch 8*

**Inference.** The process of using a trained model to make predictions on new, unseen data. Distinct from training, where the model learns from data. *Ch 18*

**Instruction Tuning.** Fine-tuning a language model on a dataset of (instruction, response) pairs to improve its ability to follow diverse instructions at inference time. *Ch 8*

**Jacobian.** A matrix of all first-order partial derivatives of a vector-valued function. Important in understanding how transformations propagate through neural network layers. *Ch 3*

**KV Cache (Key-Value Cache).** A mechanism in autoregressive Transformer inference that stores previously computed key and value tensors to avoid redundant computation at each decoding step. *Ch 18*

**L1 / L2 Regularization.** Techniques that add a penalty term to the loss function proportional to the absolute value (L1) or square (L2) of model parameters. L1 encourages sparsity; L2 encourages small weights. *Ch 5*

**Latent Space.** The lower-dimensional space of learned representations within a model, where data points are encoded as continuous vectors that capture meaningful structure. *Ch 17*

**Layer Normalization.** A normalization technique that normalizes activations across the feature dimension of a single training example, rather than across the batch. Standard in Transformers. *Ch 7*

**Learning Rate.** A hyperparameter that controls the step size of parameter updates during optimization. Too high causes divergence; too low causes slow convergence. *Ch 3*

**Learning Rate Schedule.** A strategy for adjusting the learning rate during training, such as cosine decay, linear warmup, or step decay. *Ch 6*

**Linear Regression.** A supervised learning model that predicts a continuous target as a linear combination of input features. The simplest and most fundamental regression model. *Ch 5*

**Logistic Regression.** A supervised learning model for binary classification that applies a sigmoid function to a linear combination of features. Despite its name, it is a classification algorithm. *Ch 5*

**LoRA (Low-Rank Adaptation).** A parameter-efficient fine-tuning method that freezes the pretrained weights and injects trainable low-rank matrices into each layer, dramatically reducing the number of trainable parameters. *Ch 8*

**Loss Function.** A function that quantifies the discrepancy between a model's predictions and the true values. Training minimizes this function. Common examples include cross-entropy, MSE, and contrastive loss. *Ch 3, 5*

**Matrix Decomposition.** Factoring a matrix into a product of simpler matrices. Common decompositions include SVD, eigendecomposition, QR, and Cholesky. Fundamental to many ML algorithms. *Ch 2*

**Metric Learning.** Training a model to learn a distance function (metric) such that similar examples are close and dissimilar examples are far apart in the learned embedding space. *Ch 22*

**Mini-Batch.** A subset of the training data used in one iteration of stochastic gradient descent. Mini-batch training balances the noise of SGD with the efficiency of full-batch training. *Ch 3*

**Mixed Precision Training.** A training technique that uses lower-precision floating-point formats (fp16 or bf16) for most computations while maintaining fp32 for critical operations, reducing memory usage and increasing throughput. *Ch 13*

**MLOps.** The practice of applying DevOps principles to machine learning systems, encompassing model versioning, deployment, monitoring, and lifecycle management. *Ch 19*

**Model Parallelism.** A distributed training strategy where different parts of a model are placed on different devices. Includes tensor parallelism (splitting individual layers) and pipeline parallelism (splitting layer sequences). *Ch 12*

**Model Registry.** A centralized repository for managing ML model versions, metadata, and deployment stages. Tools include MLflow Model Registry and cloud-specific offerings. *Ch 19*

**Monte Carlo Method.** A computational technique that uses repeated random sampling to obtain numerical estimates of quantities that may be difficult to compute analytically. *Ch 4*

**Multi-Head Attention.** An attention mechanism that runs multiple attention operations in parallel (heads), each with different learned projections, then concatenates and projects the results. Core component of Transformers. *Ch 7*

**Multi-Query Attention (MQA).** An attention variant where all attention heads share a single set of key-value projections, significantly reducing memory bandwidth requirements during inference. *Ch 7*

**Multimodal Model.** A model that can process and reason across multiple data modalities (e.g., text, images, audio, video) within a unified architecture. *Ch 17*

**NCCL (NVIDIA Collective Communications Library).** NVIDIA's library for multi-GPU and multi-node collective communication operations (all-reduce, broadcast, etc.). The standard communication backend for distributed PyTorch training. *Ch 12*

**Normalization.** Techniques for rescaling data or activations to have specific statistical properties (e.g., zero mean, unit variance). Includes feature normalization, batch normalization, layer normalization, and RMS normalization. *Ch 2, 7*

**Overfitting.** A condition where a model learns the training data too well, including noise and outliers, resulting in poor generalization to unseen data. *Ch 5*

**PagedAttention.** A memory management technique used in vLLM that manages KV cache memory using paging (inspired by operating system virtual memory), enabling efficient batching of variable-length sequences during LLM inference. *Ch 18*

**PCA (Principal Component Analysis).** A dimensionality reduction technique that projects data onto the directions of maximum variance. Based on eigendecomposition of the covariance matrix. *Ch 2, 5*

**Perplexity.** A metric for evaluating language models, defined as the exponentiated average negative log-likelihood per token. Lower perplexity indicates better prediction. *Ch 10*

**Pipeline Parallelism.** A distributed training strategy that partitions a model's layers across multiple devices and processes different micro-batches simultaneously, similar to an assembly line. *Ch 12*

**Positional Encoding.** A mechanism for injecting information about the position of tokens in a sequence into a Transformer model, since the attention mechanism itself is permutation-invariant. Common implementations include sinusoidal, learned, and rotary (RoPE) encodings. *Ch 7*

**Pretraining.** The initial phase of training a large model on a broad dataset (typically unsupervised) to learn general representations before fine-tuning on specific tasks. *Ch 10*

**Prompt Engineering.** The practice of designing and optimizing the text prompt given to a language model to elicit desired behavior without modifying the model's weights. *Ch 15*

**QLoRA (Quantized LoRA).** A fine-tuning method that combines 4-bit quantization of the base model with LoRA adapters, enabling fine-tuning of very large models on consumer GPUs. *Ch 8*

**Quantization.** The process of reducing the precision of model weights and/or activations (e.g., from fp32 to int8 or int4) to decrease model size and increase inference speed, with minimal loss in quality. *Ch 18*

**RAG (Retrieval-Augmented Generation).** A technique that enhances LLM outputs by retrieving relevant information from an external knowledge base and including it in the prompt context before generation. *Ch 15*

**Random Forest.** An ensemble learning method that trains multiple decision trees on random subsets of data and features, then aggregates their predictions. Robust and widely used for tabular data. *Ch 5*

**ReAct.** A prompting framework that interleaves reasoning traces (chain-of-thought) with actions (tool calls), enabling LLMs to plan and execute multi-step tasks. *Ch 16*

**Recall.** The proportion of actual positive cases that were correctly identified by the model. Recall = TP / (TP + FN). Important when the cost of missing positives is high. *Ch 5*

**Regression.** A supervised learning task where the model predicts a continuous numerical value. *Ch 5*

**Regularization.** Techniques used to prevent overfitting by constraining the model's complexity. Examples include L1/L2 penalties, dropout, early stopping, and data augmentation. *Ch 5, 7*

**Reinforcement Learning (RL).** A learning paradigm where an agent learns to make decisions by taking actions in an environment to maximize cumulative reward, receiving feedback as rewards or penalties. *Ch 23*

**Reinforcement Learning from Human Feedback (RLHF).** A technique for aligning language models with human preferences by training a reward model on human comparisons and then optimizing the language model using reinforcement learning (typically PPO). *Ch 9*

**Residual Connection (Skip Connection).** A connection that adds the input of a layer directly to its output, enabling gradients to flow through deep networks without degradation. Introduced by ResNet. *Ch 7*

**RMSNorm (Root Mean Square Layer Normalization).** A simplified variant of layer normalization that normalizes using only the root mean square of activations, omitting the mean-centering step. Adopted by LLaMA and many modern LLMs. *Ch 7*

**RNN (Recurrent Neural Network).** A neural network architecture designed for sequential data, where hidden states are passed from one time step to the next. Includes variants such as LSTM and GRU. *Ch 7*

**ROC Curve (Receiver Operating Characteristic).** A plot of the true positive rate against the false positive rate at various classification thresholds. The area under the ROC curve (AUC-ROC) summarizes classifier performance. *Ch 5*

**RoPE (Rotary Position Embedding).** A positional encoding method that encodes position information by rotating the query and key vectors, enabling relative position awareness and extrapolation to longer sequences. Standard in modern LLMs. *Ch 7*

**Scaling Laws.** Empirical relationships describing how model performance (typically measured as loss) changes as a function of model size, dataset size, and compute budget. *Ch 11*

**Self-Attention.** An attention mechanism where queries, keys, and values all come from the same sequence, allowing each position to attend to every other position. The core operation in Transformer models. *Ch 7*

**Self-Supervised Learning.** A learning paradigm where the model generates its own supervision signal from unlabeled data, typically by predicting masked or future parts of the input. Pretraining of LLMs is self-supervised. *Ch 10*

**SentencePiece.** A language-independent unsupervised text tokenizer and detokenizer that directly learns subword units from raw text. Used by many LLMs including LLaMA. *Ch 10*

**SGD (Stochastic Gradient Descent).** A variant of gradient descent that updates parameters using the gradient computed on a single example or a small mini-batch, rather than the entire dataset. *Ch 3*

**Softmax.** A function that converts a vector of real numbers into a probability distribution, where each output is in [0, 1] and all outputs sum to 1. Used in the output layer of classifiers and in attention mechanisms. *Ch 7*

**Speculative Decoding.** An inference optimization technique where a smaller, faster "draft" model generates candidate tokens that are then verified in parallel by the larger "target" model, accelerating autoregressive generation. *Ch 18*

**SVM (Support Vector Machine).** A supervised learning algorithm that finds the hyperplane that maximally separates classes in feature space. Can handle non-linear boundaries via the kernel trick. *Ch 5*

**t-SNE (t-Distributed Stochastic Neighbor Embedding).** A dimensionality reduction technique optimized for visualization, preserving local neighborhood structure of high-dimensional data in two or three dimensions. *Ch 5*

**Temperature.** A hyperparameter used during text generation that scales the logits before applying softmax. Higher temperature increases randomness; lower temperature makes outputs more deterministic. *Ch 18*

**Tensor.** A multi-dimensional array that generalizes scalars (0D), vectors (1D), and matrices (2D) to arbitrary dimensions. The fundamental data structure in deep learning frameworks. *Ch 2, 6*

**Tensor Parallelism.** A model parallelism strategy that splits individual layers (typically large matrix multiplications) across multiple devices, enabling layers too large for a single device. *Ch 12*

**TF-IDF (Term Frequency-Inverse Document Frequency).** A numerical statistic that reflects the importance of a word in a document relative to a corpus. A classical text representation technique, now largely superseded by neural embeddings. *Ch 5*

**Tokenization.** The process of converting raw text into a sequence of discrete tokens (words, subwords, or characters) that serve as input to a language model. *Ch 10*

**Top-k Sampling.** A text generation strategy that restricts sampling to the k highest-probability tokens at each step, introducing controlled randomness while avoiding low-probability tokens. *Ch 18*

**Top-p (Nucleus) Sampling.** A text generation strategy that samples from the smallest set of tokens whose cumulative probability exceeds p, adapting the number of candidates to the model's confidence at each step. *Ch 18*

**Transfer Learning.** The practice of applying knowledge learned from one task or domain to a different but related task or domain. Pretraining followed by fine-tuning is the dominant form in modern ML. *Ch 8*

**Transformer.** A neural network architecture based entirely on self-attention mechanisms, without recurrence or convolution. Introduced in the "Attention Is All You Need" paper (2017). The foundation of modern LLMs and increasingly used across all modalities. *Ch 7*

**Underfitting.** A condition where a model is too simple to capture the underlying patterns in the data, resulting in poor performance on both training and test data. *Ch 5*

**Vanishing Gradient Problem.** A difficulty in training deep networks where gradients become exponentially small as they are backpropagated through many layers, effectively stopping learning in early layers. Addressed by residual connections, LSTMs, and normalization. *Ch 7*

**Variance (Statistical).** A measure of how much a model's predictions vary across different training sets. High variance leads to overfitting. *Ch 4, 5*

**Vector Database.** A database optimized for storing, indexing, and querying high-dimensional vector embeddings. Used in RAG systems for efficient similarity search. Examples include Pinecone, Weaviate, ChromaDB, and FAISS. *Ch 15*

**ViT (Vision Transformer).** A model architecture that applies the Transformer directly to sequences of image patches, demonstrating that pure attention-based models can match or exceed CNNs on image classification. *Ch 22*

**Weight Decay.** A regularization technique that adds a term proportional to the squared magnitude of the weights to the loss function, encouraging smaller weights. In AdamW, this is decoupled from the gradient update. *Ch 3*

**Zero-Shot Learning.** A learning paradigm where a model performs a task it was not explicitly trained on, using only a natural language description of the task. LLMs exhibit zero-shot capabilities through instruction following. *Ch 8*

**ZeRO (Zero Redundancy Optimizer).** A memory optimization technique that partitions optimizer states, gradients, and parameters across devices in distributed training, eliminating memory redundancy. Implemented in DeepSpeed and PyTorch FSDP. *Ch 12*

---

*Cross-references (Ch N) indicate the chapter where the term is first introduced or most thoroughly discussed. Many terms appear in multiple chapters; only the primary reference is listed.*
