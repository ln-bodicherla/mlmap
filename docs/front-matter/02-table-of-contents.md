# Table of Contents

---

## Front Matter
- Title Page
- Preface
- Table of Contents
- How to Use This Book

---

## PART I: FOUNDATIONS

### Chapter 1: Python Mastery for Machine Learning
1.1 Core Language Features and Idioms
1.2 Data Structures — Lists, Dicts, Sets, and Beyond
1.3 Comprehensions and Generator Expressions
1.4 Functions — Decorators, Closures, and Lambdas
1.5 Object-Oriented Python — Classes, Inheritance, and Dataclasses
1.6 Context Managers and Resource Handling
1.7 Concurrency — Threading, Multiprocessing, and Asyncio
1.8 Memory Management and Profiling
1.9 Type Hints and Static Analysis with Mypy
1.10 Testing with Pytest
1.11 Packaging and Environment Management
1.12 NumPy — The Foundation of Numerical Python
1.13 Pandas — Data Manipulation at Scale
1.14 Polars — The Modern Alternative
1.15 Visualization — Matplotlib and Seaborn
1.16 Performance — Cython and Numba
References

### Chapter 2: Linear Algebra for Machine Learning
2.1 Vectors — Geometry, Operations, and Intuition
2.2 Matrices — Operations, Types, and Properties
2.3 Systems of Linear Equations
2.4 Vector Spaces and Subspaces
2.5 Linear Transformations
2.6 Eigenvalues and Eigenvectors
2.7 Singular Value Decomposition
2.8 Matrix Norms and Regularization
2.9 Tensors and Einstein Summation
2.10 Linear Algebra in Practice — NumPy and PyTorch
2.11 Applications — PCA, Recommendation Systems, and LoRA
References

### Chapter 3: Calculus and Optimization
3.1 Derivatives and the Chain Rule
3.2 Partial Derivatives and Gradients
3.3 Backpropagation — Calculus of Neural Networks
3.4 Gradient Descent and Its Variants
3.5 Momentum, RMSProp, Adam, and AdamW
3.6 Learning Rate Scheduling
3.7 Jacobians and Hessians
3.8 Convex vs. Non-Convex Optimization
3.9 Numerical Stability — Log-Sum-Exp and Softmax
3.10 Constrained Optimization and Lagrange Multipliers
3.11 Building Backpropagation from Scratch
References

### Chapter 4: Probability and Statistics
4.1 Probability Fundamentals — Axioms and Rules
4.2 Conditional Probability and Bayes' Theorem
4.3 Common Probability Distributions
4.4 Expectation, Variance, and Moments
4.5 Maximum Likelihood Estimation
4.6 Maximum A Posteriori Estimation
4.7 Information Theory — Entropy, KL Divergence, and Cross-Entropy
4.8 Hypothesis Testing for ML Engineers
4.9 Bayesian Inference and MCMC
4.10 Probabilistic Graphical Models — HMMs and CRFs
4.11 Statistical Foundations of Loss Functions
References

---

## PART II: CLASSICAL MACHINE LEARNING & CORE FRAMEWORKS

### Chapter 5: Classical Machine Learning with Scikit-Learn
5.1 The Machine Learning Pipeline
5.2 Linear Regression — From Simple to Regularized
5.3 Logistic Regression — Classification Fundamentals
5.4 Support Vector Machines — Margins and Kernels
5.5 Decision Trees — Splitting Criteria and Pruning
5.6 Ensemble Methods — Bagging, Boosting, and Stacking
5.7 XGBoost, LightGBM, and CatBoost
5.8 Clustering — K-Means, DBSCAN, and Hierarchical Methods
5.9 Dimensionality Reduction — PCA, t-SNE, and UMAP
5.10 Feature Engineering — The Art and Science
5.11 Model Evaluation — Metrics, Cross-Validation, and Calibration
5.12 Handling Class Imbalance
5.13 Hyperparameter Optimization — Grid Search to Bayesian Methods
5.14 AutoML — When and How to Use It
References

### Chapter 6: Deep Learning with PyTorch
6.1 Tensors — Creation, Operations, and GPU Acceleration
6.2 Autograd — Automatic Differentiation
6.3 Building Neural Networks with nn.Module
6.4 Loss Functions — A Comprehensive Guide
6.5 Optimizers — SGD to AdamW
6.6 Custom Datasets and DataLoaders
6.7 Training Loops — Best Practices
6.8 Regularization — Dropout, BatchNorm, Weight Decay
6.9 Learning Rate Scheduling — Warmup, Cosine Decay, OneCycleLR
6.10 Gradient Clipping and Training Stability
6.11 torch.compile — PyTorch 2.0 Graph Compilation
6.12 Profiling with torch.profiler
6.13 Custom CUDA Extensions
References

### Chapter 7: Neural Network Architectures
7.1 Multi-Layer Perceptrons — Universal Approximation
7.2 Convolutional Neural Networks — Convolution, Pooling, and BatchNorm
7.3 Landmark CNN Architectures — AlexNet to EfficientNet
7.4 Recurrent Neural Networks — RNNs, LSTMs, and GRUs
7.5 The Attention Mechanism
7.6 The Transformer Architecture
7.7 Vision Transformers
7.8 State Space Models — S4 and Mamba
7.9 Mixture of Experts — Efficient Scaling
7.10 RWKV — Linear Attention
7.11 Architecture Design Principles
References

---

## PART III: LARGE LANGUAGE MODELS

### Chapter 8: Fine-Tuning and Adaptation
8.1 Transfer Learning — Why Fine-Tuning Works
8.2 Full Fine-Tuning — When and How
8.3 LoRA — Low-Rank Adaptation
8.4 QLoRA — Quantized Fine-Tuning
8.5 DoRA, IA3, and Other PEFT Methods
8.6 Prefix Tuning and Prompt Tuning
8.7 Mixed Precision Training — FP16, BF16, and Beyond
8.8 Gradient Checkpointing — Memory vs. Compute
8.9 Continued Pretraining for Domain Adaptation
8.10 Practical Fine-Tuning — A Complete Walkthrough
References

### Chapter 9: Alignment — From RLHF to DPO
9.1 The Alignment Problem
9.2 Instruction Tuning at Scale
9.3 Reward Modeling
9.4 RLHF — Reinforcement Learning from Human Feedback
9.5 DPO — Direct Preference Optimization
9.6 Constitutional AI
9.7 Rejection Sampling and Best-of-N
9.8 Building Instruction Datasets
9.9 Evaluation of Aligned Models
References

### Chapter 10: LLM Pretraining
10.1 The Pretraining Paradigm
10.2 Data Collection and Curation
10.3 Deduplication — MinHash and Beyond
10.4 Quality Filtering and Data Cleaning
10.5 Tokenization — BPE, WordPiece, and SentencePiece
10.6 Training Objectives — Causal LM vs. Masked LM
10.7 Training Stability — Loss Spikes, Z-Loss, and Recovery
10.8 Data Mixing and Curriculum Learning
10.9 Checkpoint Management and Averaging
10.10 Modern Optimizers — AdamW, Lion, and Muon
10.11 Training a Small Language Model from Scratch
References

### Chapter 11: Scaling Laws and Emergent Abilities
11.1 Neural Scaling Laws — The Kaplan Framework
11.2 Chinchilla Optimal Training
11.3 Emergent Capabilities at Scale
11.4 Inference-Time Compute Scaling
11.5 The Bitter Lesson and Its Implications
11.6 Compute-Optimal Strategies for Practitioners
References

---

## PART IV: DISTRIBUTED TRAINING & PERFORMANCE

### Chapter 12: Parallelism Strategies
12.1 Why Distributed Training Is Necessary
12.2 Data Parallelism and PyTorch DDP
12.3 Gradient Accumulation
12.4 Fully Sharded Data Parallel (FSDP)
12.5 DeepSpeed ZeRO — Stages 1, 2, and 3
12.6 ZeRO-Infinity — CPU and NVMe Offloading
12.7 Tensor Parallelism
12.8 Pipeline Parallelism
12.9 Sequence Parallelism
12.10 Expert Parallelism for MoE Models
12.11 3D Parallelism — Putting It All Together
References

### Chapter 13: Mixed Precision and Memory Optimization
13.1 Number Formats — FP32, FP16, BF16, INT8, INT4
13.2 Mixed Precision Training in Practice
13.3 Flash Attention — Versions 1, 2, and 3
13.4 Quantization for Inference — GPTQ, AWQ, and GGUF
13.5 Speculative Decoding
13.6 PagedAttention and vLLM Memory Management
13.7 Continuous Batching for Serving
13.8 KV Cache Optimization
References

### Chapter 14: Profiling and Performance Tuning
14.1 Understanding GPU Architecture
14.2 Compute-Bound vs. Memory-Bound Operations
14.3 Model FLOP Utilization (MFU)
14.4 PyTorch Profiler — A Complete Guide
14.5 NVIDIA Nsight Systems and Compute
14.6 Custom Triton Kernels
14.7 CUDA Programming Fundamentals
14.8 End-to-End Performance Optimization Case Study
References

---

## PART V: GENERATIVE AI & ADVANCED APPLICATIONS

### Chapter 15: RAG and Retrieval Systems
15.1 The RAG Paradigm — Architecture and Motivation
15.2 Document Processing and Chunking Strategies
15.3 Embedding Models — From Word2Vec to Modern Encoders
15.4 Vector Databases — FAISS, Pinecone, ChromaDB, and Weaviate
15.5 Retrieval Strategies — Dense, Sparse, and Hybrid
15.6 Reranking — Cross-Encoders and ColBERT
15.7 Advanced RAG — HyDE, GraphRAG, and Multimodal RAG
15.8 Evaluation with RAGAS
15.9 Long-Context Models vs. Retrieval
15.10 Production RAG — Architecture Patterns
References

### Chapter 16: Agentic AI and Multi-Agent Systems
16.1 From Chatbots to Agents
16.2 The ReAct Framework
16.3 Tool Calling and Function Use
16.4 LangChain and LangGraph
16.5 AutoGen and Multi-Agent Conversations
16.6 Planning Agents — Tree of Thought and MCTS
16.7 Memory Systems for Agents
16.8 Computer Use and Code Execution Agents
16.9 Multi-Agent Coordination Patterns
16.10 MCP — Model Context Protocol
16.11 Evaluating Agent Systems
References

### Chapter 17: Multimodal Models
17.1 The Vision-Language Connection — CLIP
17.2 Image Generation — GANs to Diffusion Models
17.3 Stable Diffusion — Architecture and Sampling
17.4 Vision-Language Models — BLIP-2, LLaVA, and GPT-4V
17.5 Document Understanding — LayoutLM and Donut
17.6 Video Understanding
17.7 Audio and Speech — Whisper and Beyond
17.8 Unified Multimodal Architectures
References

### Chapter 18: LLM Serving and Inference
18.1 The Serving Challenge — Latency, Throughput, and Cost
18.2 BentoML — Model Packaging and Serving
18.3 vLLM — High-Throughput LLM Serving
18.4 Text Generation Inference (TGI)
18.5 ONNX and TensorRT Optimization
18.6 Triton Inference Server
18.7 Quantization Formats for Deployment
18.8 LoRA Serving — Multi-Tenant Deployments
18.9 Structured Generation with SGLang
18.10 Cost Optimization Strategies
References

---

## PART VI: MLOps & INFRASTRUCTURE

### Chapter 19: MLOps and Experiment Management
19.1 The MLOps Lifecycle
19.2 Experiment Tracking — MLflow and W&B
19.3 Data Version Control with DVC
19.4 Pipeline Orchestration — Airflow, Prefect, and Dagster
19.5 Kubeflow Pipelines
19.6 Ray — Distributed Python for ML
19.7 Metaflow — Netflix's ML Platform
19.8 Model Monitoring — Drift Detection and Alerting
19.9 CI/CD for Machine Learning
19.10 Containerization and Kubernetes for ML
References

### Chapter 20: Data Engineering for ML
20.1 The Modern Data Stack
20.2 Apache Spark and PySpark
20.3 Real-Time Streaming with Apache Kafka
20.4 Delta Lake and the Lakehouse Architecture
20.5 dbt — Analytics Engineering
20.6 Apache Iceberg and Open Table Formats
20.7 Apache Flink — True Stream Processing
20.8 Feature Stores — Feast and Beyond
20.9 Data Quality — Great Expectations and Data Contracts
20.10 Federated Query Engines — Trino and Presto
References

### Chapter 21: Cloud Platforms for ML
21.1 Choosing a Cloud Platform
21.2 AWS for ML — SageMaker, Bedrock, and Beyond
21.3 Google Cloud for ML — Vertex AI and TPUs
21.4 Azure for ML — Azure ML and OpenAI Service
21.5 Infrastructure as Code — Terraform and CloudFormation
21.6 Multi-Cloud and Hybrid Strategies
21.7 Cost Management for ML Workloads
References

---

## PART VII: FRONTIER TOPICS & SPECIALIZATIONS

### Chapter 22: Advanced Computer Vision
22.1 Vision Transformers — ViT and Swin
22.2 Object Detection — YOLO to DETR
22.3 Instance and Semantic Segmentation
22.4 3D Vision — NeRF and Gaussian Splatting
22.5 Video Understanding and Action Recognition
22.6 Pose Estimation
22.7 OCR and Document AI at Scale
22.8 Depth Estimation
References

### Chapter 23: Reinforcement Learning
23.1 Markov Decision Processes
23.2 Value-Based Methods — Q-Learning and DQN
23.3 Policy Gradient Methods — REINFORCE to PPO
23.4 Actor-Critic Methods
23.5 Multi-Armed Bandits
23.6 Model-Based Reinforcement Learning
23.7 Offline Reinforcement Learning
23.8 RLHF — RL for Language Model Alignment
References

### Chapter 24: Graph Machine Learning
24.1 Graph Fundamentals and Representations
24.2 Graph Neural Networks — GCN, GraphSAGE, and GAT
24.3 PyTorch Geometric
24.4 Graph Transformers
24.5 Knowledge Graphs and Embeddings
24.6 Temporal and Dynamic Graphs
24.7 Molecular Graphs and Drug Discovery
24.8 Graph Generation
References

### Chapter 25: Time Series and Forecasting
25.1 Time Series Fundamentals
25.2 Classical Methods — ARIMA, SARIMA, and Exponential Smoothing
25.3 Machine Learning for Time Series
25.4 Deep Learning — TCN, N-BEATS, and PatchTST
25.5 Foundation Models for Time Series
25.6 Anomaly Detection at Scale
25.7 Conformal Prediction — Uncertainty Quantification
References

### Chapter 26: ML Security and Responsible AI
26.1 Model Explainability — SHAP, LIME, and Beyond
26.2 Fairness in Machine Learning
26.3 Adversarial Attacks and Robustness
26.4 Data Poisoning and Backdoor Attacks
26.5 LLM Safety — Jailbreaking and Prompt Injection
26.6 Differential Privacy
26.7 Federated Learning
26.8 Model Watermarking
26.9 Building Trustworthy AI Systems
References

---

## Appendices

### Appendix A: Recommended Study Plan and Timelines
### Appendix B: Essential Papers Reading List
### Appendix C: Tools and Libraries Quick Reference
### Appendix D: Complete Bibliography
### Appendix E: Glossary of Terms

---

**Index**

