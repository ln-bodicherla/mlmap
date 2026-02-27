# Appendix C: Tools and Libraries Quick Reference

This appendix provides a categorized reference of the tools, libraries, and frameworks used throughout the book. For each entry, we include a brief description, installation command, and the primary use case relevant to this book. All installation commands assume a Unix-like environment with Python 3.10+ and pip.

> **Note:** Version numbers are omitted intentionally. Always install the latest stable release unless a specific version is required for compatibility. Pin versions in production using a `requirements.txt` or `pyproject.toml`.

---

## 1. Python Environment and Package Management

| Tool | Description | Install | Key Use Case |
|------|-------------|---------|-------------|
| **Python** | Primary programming language for ML | System-specific | All chapters |
| **conda** | Environment and package manager for data science | `curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh` | Isolated environments with CUDA support |
| **pip** | Python package installer | Included with Python | Installing Python packages |
| **uv** | Fast Python package installer and resolver | `pip install uv` | Drop-in pip replacement with 10--100x faster installs |
| **pyenv** | Python version management | `curl https://pyenv.run \| bash` | Managing multiple Python versions |
| **poetry** | Dependency management and packaging | `pip install poetry` | Project dependency management with lock files |
| **venv** | Built-in virtual environment module | Included with Python | Lightweight isolated environments |

---

## 2. Mathematics and Scientific Computing

| Tool | Description | Install | Key Use Case |
|------|-------------|---------|-------------|
| **NumPy** | Fundamental array computing library | `pip install numpy` | Matrix operations, linear algebra (Ch 2--4) |
| **SciPy** | Scientific computing library | `pip install scipy` | Optimization, statistics, signal processing (Ch 3--4) |
| **SymPy** | Symbolic mathematics | `pip install sympy` | Symbolic differentiation, verifying calculus derivations (Ch 3) |
| **JAX** | Composable transformations of NumPy programs | `pip install jax jaxlib` | Automatic differentiation, JIT compilation, research code |
| **Einops** | Flexible tensor operations with Einstein notation | `pip install einops` | Readable tensor reshaping in deep learning code (Ch 7) |

---

## 3. Data Processing and Analysis

| Tool | Description | Install | Key Use Case |
|------|-------------|---------|-------------|
| **pandas** | Data analysis and manipulation | `pip install pandas` | Tabular data loading, cleaning, exploration (Ch 1, 5) |
| **Polars** | Fast DataFrame library written in Rust | `pip install polars` | High-performance data processing for large datasets (Ch 20) |
| **Apache Spark (PySpark)** | Distributed data processing | `pip install pyspark` | Large-scale data pipelines (Ch 20) |
| **Dask** | Parallel computing with familiar APIs | `pip install dask[complete]` | Scaling pandas/NumPy workflows beyond memory (Ch 20) |
| **Apache Arrow** | Columnar in-memory data format | `pip install pyarrow` | Efficient data interchange between systems (Ch 20) |
| **DuckDB** | In-process analytical database | `pip install duckdb` | SQL analytics on local files without a server (Ch 20) |

---

## 4. Data Validation and Quality

| Tool | Description | Install | Key Use Case |
|------|-------------|---------|-------------|
| **Great Expectations** | Data validation framework | `pip install great-expectations` | Data quality checks in ML pipelines (Ch 19--20) |
| **Pydantic** | Data validation using Python type hints | `pip install pydantic` | Input/output validation for ML services (Ch 18--19) |
| **pandera** | Statistical data validation for pandas | `pip install pandera` | Schema validation for DataFrames (Ch 20) |

---

## 5. Classical Machine Learning

| Tool | Description | Install | Key Use Case |
|------|-------------|---------|-------------|
| **scikit-learn** | Comprehensive ML library | `pip install scikit-learn` | Classification, regression, clustering, evaluation (Ch 5) |
| **XGBoost** | Gradient boosting framework | `pip install xgboost` | High-performance gradient boosted trees (Ch 5) |
| **LightGBM** | Fast gradient boosting by Microsoft | `pip install lightgbm` | Large-scale gradient boosting (Ch 5) |
| **CatBoost** | Gradient boosting with categorical support | `pip install catboost` | Datasets with many categorical features (Ch 5) |
| **Optuna** | Hyperparameter optimization framework | `pip install optuna` | Automated hyperparameter tuning (Ch 5) |
| **SHAP** | SHapley Additive exPlanations | `pip install shap` | Model interpretability and feature importance (Ch 5) |

---

## 6. Deep Learning Frameworks

| Tool | Description | Install | Key Use Case |
|------|-------------|---------|-------------|
| **PyTorch** | Deep learning framework | `pip install torch torchvision torchaudio` | Primary DL framework throughout the book (Ch 6--7, 12--14) |
| **PyTorch Lightning** | High-level PyTorch wrapper | `pip install lightning` | Reducing boilerplate in training loops (Ch 6) |
| **TensorFlow** | Deep learning framework by Google | `pip install tensorflow` | Alternative DL framework; Keras API |
| **Hugging Face Transformers** | Pretrained model library | `pip install transformers` | Loading and fine-tuning pretrained models (Ch 8--11) |
| **Hugging Face Datasets** | Dataset loading library | `pip install datasets` | Standardized dataset loading and processing (Ch 8) |
| **Hugging Face Accelerate** | Distributed training abstraction | `pip install accelerate` | Multi-GPU and multi-node training (Ch 12) |
| **timm** | PyTorch image models collection | `pip install timm` | Pretrained vision models and architectures (Ch 22) |
| **torchmetrics** | Modular metric computation | `pip install torchmetrics` | Standardized metric computation during training |

---

## 7. Large Language Model Tools

| Tool | Description | Install | Key Use Case |
|------|-------------|---------|-------------|
| **Hugging Face TRL** | Transformer Reinforcement Learning | `pip install trl` | RLHF, DPO, and SFT training (Ch 9) |
| **Hugging Face PEFT** | Parameter-Efficient Fine-Tuning | `pip install peft` | LoRA, QLoRA, and other PEFT methods (Ch 8) |
| **bitsandbytes** | 8-bit and 4-bit quantization | `pip install bitsandbytes` | Model quantization for efficient fine-tuning (Ch 8) |
| **vLLM** | High-throughput LLM serving | `pip install vllm` | Production LLM inference with PagedAttention (Ch 18) |
| **Text Generation Inference (TGI)** | LLM serving by Hugging Face | Docker-based deployment | Production-grade LLM serving (Ch 18) |
| **llama.cpp** | CPU/GPU inference for LLMs in C++ | Build from source | Local LLM inference, GGUF quantization (Ch 18) |
| **Ollama** | Local LLM runner | System-specific installer | Running LLMs locally for development (Ch 18) |
| **OpenAI Python SDK** | OpenAI API client | `pip install openai` | Interacting with OpenAI and compatible APIs (Ch 15--16) |
| **Anthropic Python SDK** | Anthropic API client | `pip install anthropic` | Interacting with Claude models (Ch 15--16) |
| **tiktoken** | Fast BPE tokenizer by OpenAI | `pip install tiktoken` | Token counting and tokenization (Ch 10) |
| **SentencePiece** | Unsupervised text tokenizer | `pip install sentencepiece` | Tokenizer training and inference (Ch 10) |
| **tokenizers** | Fast tokenizer library by Hugging Face | `pip install tokenizers` | High-performance tokenization (Ch 10) |

---

## 8. Retrieval-Augmented Generation (RAG)

| Tool | Description | Install | Key Use Case |
|------|-------------|---------|-------------|
| **LangChain** | LLM application framework | `pip install langchain` | Building RAG pipelines and chains (Ch 15) |
| **LlamaIndex** | Data framework for LLM applications | `pip install llama-index` | Indexing and querying document collections (Ch 15) |
| **ChromaDB** | Open-source embedding database | `pip install chromadb` | Local vector storage for RAG (Ch 15) |
| **Pinecone** | Managed vector database | `pip install pinecone-client` | Scalable vector search in production (Ch 15) |
| **Weaviate** | Open-source vector database | `pip install weaviate-client` | Vector search with hybrid capabilities (Ch 15) |
| **FAISS** | Efficient similarity search by Meta | `pip install faiss-cpu` (or `faiss-gpu`) | Fast approximate nearest neighbor search (Ch 15) |
| **Sentence-Transformers** | Sentence embedding models | `pip install sentence-transformers` | Computing document and query embeddings (Ch 15) |

---

## 9. AI Agents and Tool Use

| Tool | Description | Install | Key Use Case |
|------|-------------|---------|-------------|
| **LangGraph** | Stateful agent framework | `pip install langgraph` | Building complex multi-step agents (Ch 16) |
| **CrewAI** | Multi-agent orchestration | `pip install crewai` | Multi-agent collaborative systems (Ch 16) |
| **Instructor** | Structured outputs from LLMs | `pip install instructor` | Extracting typed data from LLM responses (Ch 16) |

---

## 10. Distributed Training and Performance

| Tool | Description | Install | Key Use Case |
|------|-------------|---------|-------------|
| **DeepSpeed** | Distributed training library by Microsoft | `pip install deepspeed` | ZeRO optimization, pipeline parallelism (Ch 12) |
| **Megatron-LM** | Large-scale training framework by NVIDIA | Clone from GitHub | Tensor and pipeline parallelism for LLMs (Ch 12) |
| **NVIDIA NCCL** | Multi-GPU communication library | System package | GPU-to-GPU communication backend (Ch 12) |
| **PyTorch FSDP** | Fully Sharded Data Parallel | Included with PyTorch | Memory-efficient distributed training (Ch 12) |
| **NVIDIA Nsight Systems** | System-wide performance profiler | NVIDIA developer tools | GPU profiling and timeline analysis (Ch 14) |
| **PyTorch Profiler** | Built-in PyTorch profiling | Included with PyTorch | Training loop profiling (Ch 14) |
| **Triton** | GPU programming language by OpenAI | `pip install triton` | Custom GPU kernels (Ch 14) |

---

## 11. MLOps and Experiment Tracking

| Tool | Description | Install | Key Use Case |
|------|-------------|---------|-------------|
| **MLflow** | ML lifecycle management platform | `pip install mlflow` | Experiment tracking, model registry (Ch 19) |
| **Weights & Biases (W&B)** | Experiment tracking and visualization | `pip install wandb` | Logging metrics, hyperparameter sweeps (Ch 19) |
| **DVC** | Data Version Control | `pip install dvc` | Versioning datasets and ML pipelines (Ch 19) |
| **Docker** | Container platform | System-specific installer | Reproducible environments, deployment (Ch 19, 21) |
| **Kubernetes** | Container orchestration | System-specific installer | Scaling ML workloads in production (Ch 21) |
| **Airflow** | Workflow orchestration | `pip install apache-airflow` | Scheduling and managing ML pipelines (Ch 19--20) |
| **Prefect** | Modern workflow orchestration | `pip install prefect` | Python-native pipeline orchestration (Ch 19) |
| **BentoML** | ML model serving framework | `pip install bentoml` | Packaging and deploying ML models (Ch 18--19) |
| **Seldon Core** | ML deployment on Kubernetes | Helm chart | Production ML serving with monitoring (Ch 19) |
| **Prometheus** | Monitoring and alerting | System-specific installer | Metrics collection for ML services (Ch 19) |
| **Grafana** | Observability dashboards | System-specific installer | Visualizing ML service metrics (Ch 19) |

---

## 12. Cloud Platforms and Services

| Tool | Description | Install | Key Use Case |
|------|-------------|---------|-------------|
| **AWS CLI** | Amazon Web Services command line | `pip install awscli` | Managing AWS resources (Ch 21) |
| **AWS SageMaker SDK** | SageMaker Python SDK | `pip install sagemaker` | Training and deploying on SageMaker (Ch 21) |
| **Google Cloud SDK** | Google Cloud command line | System-specific installer | Managing GCP resources (Ch 21) |
| **Azure CLI** | Microsoft Azure command line | System-specific installer | Managing Azure resources (Ch 21) |
| **Terraform** | Infrastructure as code | System-specific installer | Provisioning cloud infrastructure (Ch 21) |
| **Pulumi** | Infrastructure as code in Python | System-specific installer | Cloud infrastructure with familiar languages (Ch 21) |

---

## 13. Visualization

| Tool | Description | Install | Key Use Case |
|------|-------------|---------|-------------|
| **Matplotlib** | Fundamental plotting library | `pip install matplotlib` | Static plots, loss curves, data visualization (Ch 1) |
| **Seaborn** | Statistical data visualization | `pip install seaborn` | Publication-quality statistical plots (Ch 1, 5) |
| **Plotly** | Interactive plotting library | `pip install plotly` | Interactive visualizations, dashboards |
| **TensorBoard** | Training visualization by Google | `pip install tensorboard` | Monitoring training metrics and profiles (Ch 6, 14) |
| **Streamlit** | Rapid web app framework | `pip install streamlit` | Building ML demo applications (Ch 18) |
| **Gradio** | ML demo interface builder | `pip install gradio` | Quick model demos and prototypes (Ch 18) |

---

## 14. Testing and Code Quality

| Tool | Description | Install | Key Use Case |
|------|-------------|---------|-------------|
| **pytest** | Python testing framework | `pip install pytest` | Unit and integration testing for ML code (Ch 19) |
| **ruff** | Fast Python linter and formatter | `pip install ruff` | Code linting and formatting |
| **mypy** | Static type checker for Python | `pip install mypy` | Type checking ML codebases |
| **pre-commit** | Git hook framework | `pip install pre-commit` | Automated code quality checks (Ch 19) |

---

## 15. Notebook and Development Environments

| Tool | Description | Install | Key Use Case |
|------|-------------|---------|-------------|
| **Jupyter Notebook** | Interactive computing environment | `pip install notebook` | Exploratory data analysis, prototyping (Ch 1) |
| **JupyterLab** | Next-generation Jupyter interface | `pip install jupyterlab` | Full-featured notebook environment (Ch 1) |
| **VS Code** | Code editor by Microsoft | System-specific installer | Primary IDE for ML development |
| **Google Colab** | Free hosted Jupyter notebooks | Browser-based | Prototyping with free GPU access |

---

## 16. Specialized Domain Tools

### Computer Vision (Ch 22)

| Tool | Description | Install | Key Use Case |
|------|-------------|---------|-------------|
| **OpenCV** | Computer vision library | `pip install opencv-python` | Image processing, video analysis |
| **Albumentations** | Image augmentation library | `pip install albumentations` | Data augmentation for vision models |
| **Detectron2** | Object detection by Meta | `pip install detectron2` | Object detection and segmentation |
| **Ultralytics (YOLOv8)** | Real-time object detection | `pip install ultralytics` | Fast object detection and tracking |

### Reinforcement Learning (Ch 23)

| Tool | Description | Install | Key Use Case |
|------|-------------|---------|-------------|
| **Gymnasium** | RL environments (successor to Gym) | `pip install gymnasium` | Standard RL environments and API |
| **Stable-Baselines3** | Reliable RL implementations | `pip install stable-baselines3` | Training RL agents with standard algorithms |
| **RLlib (Ray)** | Scalable RL library | `pip install ray[rllib]` | Distributed RL training |

### Graph Machine Learning (Ch 24)

| Tool | Description | Install | Key Use Case |
|------|-------------|---------|-------------|
| **PyTorch Geometric (PyG)** | Graph neural network library | `pip install torch-geometric` | GNN implementation and training |
| **DGL** | Deep Graph Library | `pip install dgl` | Scalable graph neural networks |
| **NetworkX** | Graph analysis library | `pip install networkx` | Graph construction and analysis |

### Time Series (Ch 25)

| Tool | Description | Install | Key Use Case |
|------|-------------|---------|-------------|
| **statsmodels** | Statistical modeling | `pip install statsmodels` | ARIMA, exponential smoothing, statistical tests |
| **Prophet** | Forecasting by Meta | `pip install prophet` | Business time series forecasting |
| **NeuralForecast** | Neural forecasting library | `pip install neuralforecast` | Deep learning for time series |
| **GluonTS** | Probabilistic time series by Amazon | `pip install gluonts` | Probabilistic forecasting models |

---

## Installation Tips

1. **Use virtual environments.** Never install ML packages into your system Python.
2. **Match CUDA versions.** When installing PyTorch, match the CUDA version to your driver: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
3. **Use conda for complex dependencies.** Libraries with C/C++ dependencies (like FAISS-GPU, some PyG configurations) often install more cleanly with conda.
4. **Pin versions in production.** Use `pip freeze > requirements.txt` or `poetry lock` to ensure reproducibility.
5. **Check GPU availability.** After installing PyTorch, always verify: `python -c "import torch; print(torch.cuda.is_available())"`.
