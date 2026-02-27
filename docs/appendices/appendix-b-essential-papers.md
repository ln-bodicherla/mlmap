# Appendix B: Essential Papers Reading List

This appendix collects the seminal and influential papers referenced throughout the book, organized by topic area. For each paper, we provide the full citation, a brief explanation of why it matters, and a difficulty rating to help you plan your reading.

**Difficulty levels:**

- **Introductory** -- Accessible to readers with basic ML knowledge; suitable for beginners.
- **Intermediate** -- Requires solid understanding of ML fundamentals; appropriate after completing Parts I--II.
- **Advanced** -- Assumes deep knowledge of the topic; best tackled after completing the relevant book chapters.

---

## 1. Foundations of Machine Learning

### 1.1 Statistical Learning Theory

**[F1]** Vapnik, V. N. (1995). *The Nature of Statistical Learning Theory.* Springer.
- **Why it matters:** Foundational text establishing the theoretical framework for statistical learning, including VC dimension and structural risk minimization.
- **Difficulty:** Advanced

**[F2]** Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
- **Why it matters:** The definitive reference for classical statistical learning methods, bridging statistics and machine learning.
- **Difficulty:** Intermediate

**[F3]** Bishop, C. M. (2006). *Pattern Recognition and Machine Learning.* Springer.
- **Why it matters:** Comprehensive Bayesian treatment of ML; essential for understanding probabilistic models.
- **Difficulty:** Intermediate

### 1.2 Optimization

**[F4]** Bottou, L. (2010). Large-Scale Machine Learning with Stochastic Gradient Descent. *Proceedings of COMPSTAT*, 177--186.
- **Why it matters:** Provides theoretical and practical justification for SGD in large-scale learning.
- **Difficulty:** Intermediate

**[F5]** Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. *Proceedings of ICLR 2015*.
- **Why it matters:** Introduced the Adam optimizer, now the default choice for most deep learning training.
- **Difficulty:** Introductory

**[F6]** Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. *Proceedings of ICLR 2019*.
- **Why it matters:** Introduced AdamW, which corrects weight decay in Adam and is standard in modern LLM training.
- **Difficulty:** Intermediate

---

## 2. Classical Machine Learning

**[C1]** Breiman, L. (2001). Random Forests. *Machine Learning, 45*(1), 5--32.
- **Why it matters:** Introduced random forests, one of the most widely used and robust ML algorithms.
- **Difficulty:** Introductory

**[C2]** Cortes, C., & Vapnik, V. (1995). Support-Vector Networks. *Machine Learning, 20*(3), 273--297.
- **Why it matters:** Foundational paper on SVMs; introduced the kernel trick for non-linear classification.
- **Difficulty:** Intermediate

**[C3]** Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of KDD 2016*, 785--794.
- **Why it matters:** Introduced XGBoost, which dominated Kaggle competitions and industrial ML for years.
- **Difficulty:** Intermediate

**[C4]** Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. *Annals of Statistics, 29*(5), 1189--1232.
- **Why it matters:** Foundational paper on gradient boosting, the basis for XGBoost, LightGBM, and CatBoost.
- **Difficulty:** Intermediate

**[C5]** van der Maaten, L., & Hinton, G. (2008). Visualizing Data using t-SNE. *Journal of Machine Learning Research, 9*, 2579--2605.
- **Why it matters:** Introduced t-SNE, the most popular technique for visualizing high-dimensional data.
- **Difficulty:** Introductory

**[C6]** McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. *arXiv:1802.03426*.
- **Why it matters:** Introduced UMAP, a faster and often superior alternative to t-SNE for dimensionality reduction.
- **Difficulty:** Intermediate

---

## 3. Deep Learning Foundations

### 3.1 Architectures

**[D1]** Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning Representations by Back-Propagating Errors. *Nature, 323*, 533--536.
- **Why it matters:** Introduced backpropagation for training neural networks; the foundation of all modern deep learning.
- **Difficulty:** Introductory

**[D2]** LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-Based Learning Applied to Document Recognition. *Proceedings of the IEEE, 86*(11), 2278--2324.
- **Why it matters:** Introduced LeNet and convolutional neural networks for practical applications.
- **Difficulty:** Introductory

**[D3]** Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet Classification with Deep Convolutional Neural Networks. *Proceedings of NeurIPS 2012*, 1097--1105.
- **Why it matters:** AlexNet demonstrated the power of deep CNNs on ImageNet and launched the deep learning era.
- **Difficulty:** Introductory

**[D4]** He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *Proceedings of CVPR 2016*, 770--778.
- **Why it matters:** Introduced residual connections (skip connections), enabling training of very deep networks.
- **Difficulty:** Introductory

**[D5]** Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation, 9*(8), 1735--1780.
- **Why it matters:** Introduced LSTMs, solving the vanishing gradient problem for sequence modeling.
- **Difficulty:** Intermediate

**[D6]** Cho, K., van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. *Proceedings of EMNLP 2014*, 1724--1734.
- **Why it matters:** Introduced the GRU architecture and the encoder-decoder framework for sequence-to-sequence tasks.
- **Difficulty:** Intermediate

### 3.2 Training Techniques

**[D7]** Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. *Journal of Machine Learning Research, 15*, 1929--1958.
- **Why it matters:** Introduced dropout regularization, one of the most important techniques in deep learning.
- **Difficulty:** Introductory

**[D8]** Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. *Proceedings of ICML 2015*, 448--456.
- **Why it matters:** Introduced batch normalization, which dramatically stabilized and accelerated training.
- **Difficulty:** Introductory

**[D9]** Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer Normalization. *arXiv:1607.06450*.
- **Why it matters:** Introduced layer normalization, now standard in Transformer architectures.
- **Difficulty:** Introductory

**[D10]** Zhang, B., & Sennrich, R. (2019). Root Mean Square Layer Normalization. *Proceedings of NeurIPS 2019*.
- **Why it matters:** Introduced RMSNorm, widely adopted in modern LLMs (LLaMA, etc.) for its simplicity and efficiency.
- **Difficulty:** Introductory

---

## 4. Transformer Architecture and Attention

**[T1]** Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention Is All You Need. *Proceedings of NeurIPS 2017*, 5998--6008.
- **Why it matters:** Introduced the Transformer architecture; arguably the most influential ML paper of the decade.
- **Difficulty:** Intermediate

**[T2]** Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural Machine Translation by Jointly Learning to Align and Translate. *Proceedings of ICLR 2015*.
- **Why it matters:** Introduced the attention mechanism for sequence-to-sequence models.
- **Difficulty:** Intermediate

**[T3]** Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *Proceedings of NAACL-HLT 2019*, 4171--4186.
- **Why it matters:** Introduced BERT and the pre-train/fine-tune paradigm for NLP.
- **Difficulty:** Intermediate

**[T4]** Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2024). RoFormer: Enhanced Transformer with Rotary Position Embedding. *Neurocomputing, 568*, 127063.
- **Why it matters:** Introduced Rotary Position Embeddings (RoPE), now standard in most modern LLMs.
- **Difficulty:** Intermediate

**[T5]** Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Re, C. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. *Proceedings of NeurIPS 2022*.
- **Why it matters:** Introduced FlashAttention, which made Transformer training and inference significantly faster.
- **Difficulty:** Advanced

**[T6]** Shazeer, N. (2019). Fast Transformer Decoding: One Write-Head is All You Need. *arXiv:1911.02150*.
- **Why it matters:** Introduced multi-query attention (MQA), reducing memory bandwidth during inference.
- **Difficulty:** Intermediate

**[T7]** Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebron, F., & Sanghai, S. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. *Proceedings of EMNLP 2023*.
- **Why it matters:** Introduced grouped-query attention (GQA), a practical middle ground between MHA and MQA, adopted by LLaMA 2 and many others.
- **Difficulty:** Intermediate

---

## 5. Large Language Models

### 5.1 Foundational LLM Papers

**[L1]** Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018). Improving Language Understanding by Generative Pre-Training. *OpenAI Technical Report*.
- **Why it matters:** GPT-1 demonstrated that generative pretraining followed by fine-tuning yields strong NLP performance.
- **Difficulty:** Introductory

**[L2]** Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language Models are Unsupervised Multitask Learners. *OpenAI Technical Report*.
- **Why it matters:** GPT-2 showed that scaling up language models produces emergent few-shot capabilities.
- **Difficulty:** Introductory

**[L3]** Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., et al. (2020). Language Models are Few-Shot Learners. *Proceedings of NeurIPS 2020*, 1877--1901.
- **Why it matters:** GPT-3 demonstrated that sufficiently large language models can perform tasks with few or zero examples.
- **Difficulty:** Intermediate

**[L4]** Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., et al. (2023). LLaMA: Open and Efficient Foundation Language Models. *arXiv:2302.13971*.
- **Why it matters:** Released competitive open-source LLMs and catalyzed the open-source LLM ecosystem.
- **Difficulty:** Intermediate

**[L5]** Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., et al. (2023). Llama 2: Open Foundation and Fine-Tuned Chat Models. *arXiv:2307.09288*.
- **Why it matters:** Introduced Llama 2 with RLHF alignment and permissive licensing; became the foundation for many downstream models.
- **Difficulty:** Intermediate

**[L6]** Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D. S., et al. (2023). Mistral 7B. *arXiv:2310.06825*.
- **Why it matters:** Demonstrated that smaller, well-trained models can match much larger ones; introduced sliding window attention.
- **Difficulty:** Intermediate

### 5.2 Fine-Tuning and Efficiency

**[L7]** Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *Proceedings of ICLR 2022*.
- **Why it matters:** Introduced LoRA, the most widely used parameter-efficient fine-tuning method.
- **Difficulty:** Intermediate

**[L8]** Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized Language Models. *Proceedings of NeurIPS 2023*.
- **Why it matters:** Combined quantization with LoRA, enabling fine-tuning of large models on consumer GPUs.
- **Difficulty:** Intermediate

**[L9]** Wei, J., Bosma, M., Zhao, V., Guu, K., Yu, A. W., Lester, B., Du, N., Dai, A. M., & Le, Q. V. (2022). Finetuned Language Models Are Zero-Shot Learners. *Proceedings of ICLR 2022*.
- **Why it matters:** Introduced instruction tuning (FLAN), showing that fine-tuning on instructions improves zero-shot performance.
- **Difficulty:** Introductory

### 5.3 Alignment

**[L10]** Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C. L., et al. (2022). Training Language Models to Follow Instructions with Human Feedback. *Proceedings of NeurIPS 2022*.
- **Why it matters:** The InstructGPT paper; introduced RLHF for aligning language models with human preferences.
- **Difficulty:** Intermediate

**[L11]** Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. *Proceedings of NeurIPS 2023*.
- **Why it matters:** Introduced DPO, simplifying alignment by eliminating the need for a separate reward model.
- **Difficulty:** Intermediate

**[L12]** Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. *arXiv:1707.06347*.
- **Why it matters:** Introduced PPO, the RL algorithm used in most RLHF pipelines.
- **Difficulty:** Intermediate

**[L13]** Bai, Y., Jones, A., Ndousse, K., Askell, A., Chen, A., et al. (2022). Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback. *arXiv:2204.05862*.
- **Why it matters:** Detailed Anthropic's approach to RLHF, including the helpful-harmless framework.
- **Difficulty:** Intermediate

### 5.4 Scaling Laws

**[L14]** Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., & Amodei, D. (2020). Scaling Laws for Neural Language Models. *arXiv:2001.08361*.
- **Why it matters:** Established empirical scaling laws relating model size, data, and compute to loss.
- **Difficulty:** Intermediate

**[L15]** Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., et al. (2022). Training Compute-Optimal Large Language Models. *Proceedings of NeurIPS 2022*.
- **Why it matters:** The Chinchilla paper; showed that most LLMs were undertrained and provided compute-optimal scaling laws.
- **Difficulty:** Intermediate

---

## 6. Prompting and Reasoning

**[P1]** Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E., Le, Q., & Zhou, D. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. *Proceedings of NeurIPS 2022*.
- **Why it matters:** Demonstrated that chain-of-thought prompting significantly improves reasoning in LLMs.
- **Difficulty:** Introductory

**[P2]** Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. *Proceedings of ICLR 2023*.
- **Why it matters:** Introduced the ReAct framework combining reasoning traces with action steps for agentic LLM use.
- **Difficulty:** Introductory

**[P3]** Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *Proceedings of NeurIPS 2020*.
- **Why it matters:** Introduced RAG, combining retrieval with generation for knowledge-grounded outputs.
- **Difficulty:** Intermediate

**[P4]** Shinn, N., Cassano, F., Gopinath, A., Shakkottai, K., Labash, A., & Kass-Hout, T. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. *Proceedings of NeurIPS 2023*.
- **Why it matters:** Introduced self-reflection for LLM agents, enabling iterative improvement without weight updates.
- **Difficulty:** Intermediate

---

## 7. Distributed Training and Systems

**[S1]** Dean, J., Corrado, G. S., Monga, R., Chen, K., Devin, M., et al. (2012). Large Scale Distributed Deep Networks. *Proceedings of NeurIPS 2012*, 1223--1231.
- **Why it matters:** Pioneering work on distributed deep learning at Google, introducing DistBelief.
- **Difficulty:** Intermediate

**[S2]** Rajbhandari, S., Rasley, J., Ruwase, O., & He, Y. (2020). ZeRO: Memory Optimizations Toward Training Trillion Parameter Models. *Proceedings of SC 2020*.
- **Why it matters:** Introduced ZeRO (Zero Redundancy Optimizer), enabling training of models with trillions of parameters.
- **Difficulty:** Advanced

**[S3]** Shoeybi, M., Patwary, M., Puri, R., LeGresley, P., Casper, J., & Catanzaro, B. (2019). Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism. *arXiv:1909.08053*.
- **Why it matters:** Introduced efficient tensor and pipeline parallelism for training very large language models.
- **Difficulty:** Advanced

**[S4]** Micikevicius, P., Narang, S., Alben, J., Diamos, G., Elsen, E., et al. (2018). Mixed Precision Training. *Proceedings of ICLR 2018*.
- **Why it matters:** Established the foundations for mixed precision training using fp16 with loss scaling.
- **Difficulty:** Intermediate

**[S5]** Huang, Y., Cheng, Y., Bapna, A., Firat, O., Chen, M. X., et al. (2019). GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism. *Proceedings of NeurIPS 2019*.
- **Why it matters:** Introduced pipeline parallelism for training large models across multiple accelerators.
- **Difficulty:** Advanced

---

## 8. Generative Models and Multimodal

**[G1]** Goodfellow, I. J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., & Bengio, Y. (2014). Generative Adversarial Nets. *Proceedings of NeurIPS 2014*, 2672--2680.
- **Why it matters:** Introduced GANs, a fundamental generative modeling framework.
- **Difficulty:** Intermediate

**[G2]** Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. *Proceedings of NeurIPS 2020*.
- **Why it matters:** Revived diffusion models, which now power state-of-the-art image generation (Stable Diffusion, DALL-E).
- **Difficulty:** Advanced

**[G3]** Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. *Proceedings of CVPR 2022*, 10684--10695.
- **Why it matters:** Introduced latent diffusion models, making diffusion practical for high-resolution image generation.
- **Difficulty:** Advanced

**[G4]** Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. *Proceedings of ICML 2021*, 8748--8763.
- **Why it matters:** Introduced CLIP, connecting vision and language in a shared embedding space.
- **Difficulty:** Intermediate

**[G5]** Liu, H., Li, C., Wu, Q., & Lee, Y. J. (2023). Visual Instruction Tuning. *Proceedings of NeurIPS 2023*.
- **Why it matters:** Introduced LLaVA, demonstrating effective visual instruction tuning for multimodal models.
- **Difficulty:** Intermediate

---

## 9. Reinforcement Learning

**[R1]** Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., et al. (2015). Human-Level Control through Deep Reinforcement Learning. *Nature, 518*(7540), 529--533.
- **Why it matters:** Introduced deep Q-networks (DQN), demonstrating RL agents that surpass human performance on Atari games.
- **Difficulty:** Intermediate

**[R2]** Silver, D., Huang, A., Maddison, C. J., Guez, A., Sifre, L., et al. (2016). Mastering the Game of Go with Deep Neural Networks and Tree Search. *Nature, 529*(7587), 484--489.
- **Why it matters:** AlphaGo defeated a world champion Go player, demonstrating the power of deep RL combined with search.
- **Difficulty:** Intermediate

**[R3]** Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- **Why it matters:** The definitive textbook on reinforcement learning; freely available online.
- **Difficulty:** Intermediate

---

## 10. Graph Neural Networks

**[GN1]** Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. *Proceedings of ICLR 2017*.
- **Why it matters:** Introduced GCNs, the foundational architecture for graph neural networks.
- **Difficulty:** Intermediate

**[GN2]** Velickovic, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y. (2018). Graph Attention Networks. *Proceedings of ICLR 2018*.
- **Why it matters:** Introduced attention mechanisms to graph neural networks, enabling flexible neighborhood aggregation.
- **Difficulty:** Intermediate

**[GN3]** Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive Representation Learning on Large Graphs. *Proceedings of NeurIPS 2017*, 1024--1034.
- **Why it matters:** Introduced GraphSAGE, enabling inductive learning on graphs (generalization to unseen nodes).
- **Difficulty:** Intermediate

---

## 11. Time Series and Forecasting

**[TS1]** Lim, B., Arik, S. O., Loeff, N., & Pfister, T. (2021). Temporal Fusion Transformers for Interpretable Multi-Horizon Time Series Forecasting. *International Journal of Forecasting, 37*(4), 1748--1764.
- **Why it matters:** Introduced TFT, combining attention with interpretable components for time series forecasting.
- **Difficulty:** Intermediate

**[TS2]** Oreshkin, B. N., Carpov, D., Chapados, N., & Bengio, Y. (2020). N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting. *Proceedings of ICLR 2020*.
- **Why it matters:** Demonstrated that pure deep learning can outperform statistical methods on time series benchmarks.
- **Difficulty:** Intermediate

---

## 12. Responsible AI and Security

**[E1]** Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). Explaining and Harnessing Adversarial Examples. *Proceedings of ICLR 2015*.
- **Why it matters:** Introduced FGSM and provided a theoretical framework for understanding adversarial examples.
- **Difficulty:** Intermediate

**[E2]** Carlini, N., & Wagner, D. (2017). Towards Evaluating the Robustness of Neural Networks. *Proceedings of IEEE S&P 2017*, 39--57.
- **Why it matters:** Introduced the C&W attack, the gold standard for evaluating adversarial robustness.
- **Difficulty:** Advanced

**[E3]** Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., & Zhang, L. (2016). Deep Learning with Differential Privacy. *Proceedings of CCS 2016*, 308--318.
- **Why it matters:** Introduced DP-SGD for training neural networks with formal privacy guarantees.
- **Difficulty:** Advanced

**[E4]** Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T. (2019). Model Cards for Model Reporting. *Proceedings of FAT* 2019*, 220--229.
- **Why it matters:** Proposed model cards as a standardized way to document ML models, including intended use and limitations.
- **Difficulty:** Introductory

**[E5]** Bender, E. M., Gebru, T., McMillan-Major, A., & Shmitchell, S. (2021). On the Dangers of Stochastic Parrots: Can Language Models Be Too Big? *Proceedings of FAccT 2021*, 610--623.
- **Why it matters:** Influential critique of the environmental and social costs of large language models.
- **Difficulty:** Introductory

---

## Reading Order Recommendation

For readers new to the field, we recommend the following order:

1. Start with the **Introductory** papers to build intuition.
2. Read **[T1]** (Attention Is All You Need) after completing Part II of the book.
3. Read the **LLM papers [L1]--[L6]** sequentially to understand the evolution of language models.
4. Read the **alignment papers [L10]--[L13]** after Chapter 9.
5. Read the **distributed training papers [S1]--[S5]** after completing Part IV.
6. Read the **frontier topic papers** as you work through Part VII.

For experienced readers, we recommend reading the **Advanced** papers alongside the corresponding book chapters, using the chapter cross-references as a guide.
