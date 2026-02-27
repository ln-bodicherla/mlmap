# Chapter 10: LLM Pretraining

> *"All models are wrong, but some are useful."* -- George E. P. Box

## Learning Objectives

By the end of this chapter, you will be able to:

1. Explain why self-supervised next-token prediction gives rise to general-purpose language understanding and generation capabilities.
2. Design a data pipeline for LLM pretraining, including collection, deduplication, quality filtering, and tokenization.
3. Implement BPE tokenization from scratch and understand the tradeoffs between vocabulary size and sequence length.
4. Compare causal, masked, and prefix language modeling objectives and articulate when each is appropriate.
5. Diagnose and recover from training instabilities such as loss spikes and gradient explosions.
6. Implement a complete small-scale language model training pipeline from data preparation through evaluation.

---

## 10.1 The Pretraining Paradigm

The modern large language model is built on a simple but profound idea: **predict the next token**. Given a sequence of tokens $x_1, x_2, \ldots, x_{t-1}$, the model learns to predict $x_t$. The training objective is to maximize the log-likelihood of the training corpus:

$$\mathcal{L}(\theta) = \sum_{i=1}^{N} \sum_{t=1}^{T_i} \log P_\theta(x_t^{(i)} | x_1^{(i)}, \ldots, x_{t-1}^{(i)})$$

where $N$ is the number of documents and $T_i$ is the length of document $i$.

### 10.1.1 Why Next-Token Prediction Works

At first glance, next-token prediction seems too simple to produce intelligent behavior. But consider what is required to predict the next token well across a diverse corpus:

- To predict the next word in a grammatical sentence, the model must learn **syntax**.
- To predict the next word in a factual passage, the model must encode **world knowledge**.
- To predict the next step in a mathematical derivation, the model must learn **logical reasoning**.
- To predict the next line of code, the model must understand **programming semantics**.
- To predict the next word in a translated sentence, the model must learn **cross-lingual correspondences**.

In effect, next-token prediction is a **universal self-supervised objective**: any cognitive capability that helps predict the next token in any context will be learned. The loss function does not specify which capabilities to learn --- the data determines what is needed (Radford et al., 2019).

### 10.1.2 Emergent Properties

As models scale in both parameters and training data, they exhibit **emergent properties** --- capabilities that are not explicitly trained but arise as byproducts of the pretraining objective. These include:

- **In-context learning**: The ability to perform tasks specified in the prompt without any gradient updates (Brown et al., 2020).
- **Chain-of-thought reasoning**: The ability to solve multi-step problems when prompted to "think step by step" (Wei et al., 2022).
- **Translation**: Despite not being explicitly trained on parallel corpora, models learn to translate between languages present in the training data.
- **Code execution**: Models can mentally trace through code and predict outputs.

These emergent capabilities are discussed in detail in Chapter 11.

---

## 10.2 Data Collection

The quality, diversity, and scale of pretraining data are perhaps the most important determinants of model capability. Modern LLMs are trained on datasets of 1--15 trillion tokens, drawn from a variety of sources.

### 10.2.1 Common Crawl

Common Crawl is the foundation of most pretraining datasets. It is a nonprofit organization that crawls the web and makes the data freely available. Key characteristics:

- **Scale**: Petabytes of raw web data, hundreds of billions of pages.
- **Diversity**: Covers virtually every topic and language present on the web.
- **Quality**: Extremely variable --- ranges from well-edited articles to spam, boilerplate, and machine-generated content.

Processing Common Crawl into a usable pretraining dataset requires extensive filtering (Section 10.4).

**Notable processed versions:**
- **C4** (Raffel et al., 2020): Used for T5. Applied heuristic filters (language detection, deduplication, removing offensive content).
- **OSCAR**: Multilingual corpus extracted from Common Crawl with language classification.
- **RefinedWeb** (Penedo et al., 2023): Applied rigorous filtering and deduplication, producing 5 trillion tokens of high-quality web text.

### 10.2.2 Books

Book corpora provide long-form, well-edited text that teaches models extended coherence and deep topical coverage:

- **BookCorpus** (~800M tokens): Used in BERT and GPT-2. Contains ~11,000 unpublished books from smashwords.com.
- **Books3** (part of The Pile): A larger book corpus, though its use is legally contested.
- **Project Gutenberg**: Public domain books. Older texts but legally unambiguous.

### 10.2.3 Code

Code data has become essential for modern LLMs, even those not primarily intended for programming. Training on code appears to improve reasoning capabilities (Madaan et al., 2022):

- **The Stack** (Kocetkov et al., 2022): 6.4 TB of permissively licensed source code from GitHub, covering 358 programming languages.
- **StarCoder training data**: A refined version of The Stack with near-deduplication and PII removal.
- **GitHub public repositories**: The raw source, requiring extensive filtering and deduplication.

### 10.2.4 Scientific Papers

ArXiv papers, PubMed abstracts, and Semantic Scholar provide high-quality technical text:

- **ArXiv**: LaTeX source of scientific papers. Requires preprocessing to extract clean text from LaTeX markup.
- **PubMed/PubMed Central**: Biomedical literature. Provides structured abstracts and full text.
- **S2ORC**: Semantic Scholar Open Research Corpus --- 81M academic papers across all fields.

### 10.2.5 Wikipedia

Wikipedia is a small but extremely high-quality source:

- **Scale**: ~4B tokens for English Wikipedia.
- **Quality**: Well-edited, factual, encyclopedic.
- **Availability**: Free and regularly updated dumps.
- **Limitation**: Encyclopedic style only --- does not teach conversational or creative writing.

### 10.2.6 Legal and Ethical Considerations

Data collection for LLMs raises significant legal and ethical issues:

- **Copyright**: Training on copyrighted material is the subject of active litigation. The New York Times, Getty Images, and various authors have sued AI companies.
- **Privacy**: Web data contains personal information. PII (personally identifiable information) filtering is essential but imperfect.
- **Consent**: Most web authors did not consent to their work being used for AI training.
- **Licensing**: The Stack uses only permissively licensed code. Other datasets are less careful about licensing.
- **Bias**: Web data reflects societal biases. Models trained on this data will reproduce and potentially amplify these biases.

Practitioners should document their data sources, apply appropriate filtering, and stay informed about the evolving legal landscape.

---

## 10.3 Deduplication

Duplicate and near-duplicate content in the training data is a significant problem. Duplicates waste compute (the model repeatedly trains on the same content), amplify memorization (increasing the risk of regurgitating training data verbatim), and distort the learned distribution (over-represented content receives disproportionate influence).

### 10.3.1 Exact Deduplication

The simplest approach is exact deduplication: remove documents that are byte-for-byte identical.

**Implementation**: Hash each document (e.g., SHA-256) and remove duplicates:

```python
import hashlib
from collections import defaultdict

def exact_dedup(documents):
    """Remove exact duplicate documents using SHA-256 hashing."""
    seen_hashes = set()
    unique_docs = []

    for doc in documents:
        doc_hash = hashlib.sha256(doc.encode('utf-8')).hexdigest()
        if doc_hash not in seen_hashes:
            seen_hashes.add(doc_hash)
            unique_docs.append(doc)

    return unique_docs
```

For large-scale deduplication, use a Bloom filter to reduce memory usage:

```python
from pybloom_live import ScalableBloomFilter

def exact_dedup_bloom(documents, error_rate=0.001):
    """Memory-efficient exact dedup using Bloom filter."""
    bloom = ScalableBloomFilter(
        initial_capacity=1000000,
        error_rate=error_rate,
    )
    unique_docs = []

    for doc in documents:
        doc_hash = hashlib.sha256(doc.encode('utf-8')).hexdigest()
        if doc_hash not in bloom:
            bloom.add(doc_hash)
            unique_docs.append(doc)

    return unique_docs
```

Exact deduplication catches identical copies but misses near-duplicates: documents that differ by whitespace, formatting, small edits, or boilerplate additions.

### 10.3.2 Near-Deduplication with MinHash LSH

MinHash Locality-Sensitive Hashing (LSH) is the standard technique for finding near-duplicate documents at scale. It approximates Jaccard similarity between document shingle sets.

**Step 1: Shingling.** Convert each document into a set of character or word $n$-grams (shingles):

$$S(d) = \{s \mid s \text{ is a contiguous } n\text{-gram in document } d\}$$

Typically, $n = 5$--$13$ for word-level shingles.

**Step 2: MinHash signatures.** A MinHash function $h_\pi$ maps a set to a single value:

$$h_\pi(S) = \min_{s \in S} \pi(s)$$

where $\pi$ is a random permutation of the universe of shingles. The key property of MinHash is:

$$P(h_\pi(A) = h_\pi(B)) = J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

That is, the probability that two sets have the same MinHash equals their Jaccard similarity. By computing $k$ MinHash values (using $k$ different random permutations), we obtain a **signature** vector of length $k$ that compactly represents the document.

**Step 3: Locality-Sensitive Hashing.** To avoid comparing all $O(N^2)$ pairs of documents, we use LSH banding. The signature of length $k$ is divided into $b$ bands of $r$ rows each ($k = b \times r$). Two documents are considered candidate duplicates if they agree on all $r$ rows in at least one band.

The probability that two documents with Jaccard similarity $J$ are identified as candidates is:

$$P(\text{candidate}) = 1 - (1 - J^r)^b$$

This creates an S-curve: documents with similarity above the threshold $J^* \approx (1/b)^{1/r}$ are almost certainly identified, while those below are almost certainly missed.

```python
from datasketch import MinHash, MinHashLSH

def create_minhash(text, num_perm=128):
    """Create a MinHash signature for a document."""
    mh = MinHash(num_perm=num_perm)
    # Use 5-word shingles
    words = text.split()
    for i in range(len(words) - 4):
        shingle = ' '.join(words[i:i+5])
        mh.update(shingle.encode('utf-8'))
    return mh

def find_near_duplicates(documents, threshold=0.8, num_perm=128):
    """Find near-duplicate documents using MinHash LSH."""
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes = {}
    duplicates = set()

    for i, doc in enumerate(documents):
        mh = create_minhash(doc, num_perm)
        minhashes[i] = mh

        # Query for similar documents already in the index
        result = lsh.query(mh)
        if result:
            duplicates.add(i)
        else:
            lsh.insert(str(i), mh)

    return [doc for i, doc in enumerate(documents) if i not in duplicates]
```

**Practical considerations:**
- **Scale**: The `datasketch` library handles millions of documents. For billions, use the `text-dedup` library or Apache Spark-based implementations.
- **Parameters**: For web data, a Jaccard threshold of 0.7--0.8 and 128--256 MinHash permutations work well.
- **Substring deduplication**: Some approaches also remove documents that are substrings of other documents, or use suffix arrays for paragraph-level deduplication.

Lee et al. (2022) showed that deduplication improves both training efficiency and downstream performance, with diminishing returns beyond approximately 4x deduplication effort.

---

## 10.4 Quality Filtering

Even after deduplication, web-crawled data contains vast amounts of low-quality content: spam, boilerplate, machine-generated text, garbled HTML, and more. Quality filtering is essential to convert raw web data into a useful training corpus.

### 10.4.1 Perplexity Filtering

Perplexity filtering uses a reference language model (typically a small model trained on high-quality text like Wikipedia) to score documents. Documents with very high perplexity (the reference model finds them surprising) are likely low-quality.

$$\text{PPL}(x) = \exp\left(-\frac{1}{T} \sum_{t=1}^{T} \log P_{\text{ref}}(x_t | x_{<t})\right)$$

A typical approach:
1. Train a small $n$-gram or neural language model on Wikipedia.
2. Compute perplexity for each document in the web corpus.
3. Remove documents below the 5th percentile (gibberish, non-language) and above the 95th percentile (too formulaic or repetitive).

The CCNet pipeline (Wenzek et al., 2020), used to create CC-100 and subsequent datasets, is a prominent example of perplexity-based filtering.

### 10.4.2 Heuristic Filters

Practical filtering pipelines apply numerous heuristic rules. The following is representative of those used in major datasets (Penedo et al., 2023; Soldaini et al., 2024):

```python
def heuristic_quality_filter(doc):
    """Apply heuristic quality filters to a document."""
    text = doc['text']
    url = doc.get('url', '')

    # Length filter: too short or too long
    if len(text) < 100 or len(text) > 1_000_000:
        return False

    words = text.split()

    # Word count filter
    if len(words) < 50:
        return False

    # Mean word length filter (catches garbled text)
    mean_word_len = sum(len(w) for w in words) / len(words)
    if mean_word_len < 3 or mean_word_len > 10:
        return False

    # Symbol-to-word ratio (catches code dumps, garbled HTML)
    symbols = sum(1 for c in text if c in '{}[]<>|\\^~')
    if symbols / len(words) > 0.1:
        return False

    # Bullet/ellipsis ratio (catches lists, tables)
    lines = text.split('\n')
    bullet_lines = sum(1 for l in lines if l.strip().startswith(('-', '*', '•')))
    if len(lines) > 0 and bullet_lines / len(lines) > 0.9:
        return False

    # Repetition filter: fraction of repeated lines
    unique_lines = set(lines)
    if len(lines) > 0 and len(unique_lines) / len(lines) < 0.3:
        return False

    # URL blocklist (adult content, spam domains)
    blocked_domains = ['spam.com', 'casino', 'pharmacy']
    if any(domain in url for domain in blocked_domains):
        return False

    # "Lorem ipsum" and other boilerplate detection
    if 'lorem ipsum' in text.lower():
        return False

    return True
```

### 10.4.3 Classifier-Based Filtering

For maximum quality, a text classifier can be trained to distinguish "high-quality" text (e.g., Wikipedia, published books) from "low-quality" text (random web crawl):

1. Train a binary classifier: high-quality reference corpus vs. random web sample.
2. Score all documents in the training corpus.
3. Keep only documents above a quality threshold.

The LLaMA paper (Touvron et al., 2023) used a linear classifier trained on Wikipedia and random web pages, keeping only the top-scoring web documents. This approach is simple but effective, and the classifier itself can be a small fastText model trained in minutes.

---

## 10.5 Tokenization

Tokenization --- converting raw text into a sequence of discrete tokens --- is a foundational design decision that affects every aspect of training and inference.

### 10.5.1 Byte-Pair Encoding (BPE)

BPE (Sennrich et al., 2016) is the most widely used tokenization algorithm. It starts with a vocabulary of individual characters (or bytes) and iteratively merges the most frequent adjacent pair.

**Algorithm:**

```
Input: Training corpus, desired vocabulary size V
Output: Merge rules and vocabulary

1. Initialize vocabulary with all individual characters in the corpus
2. While vocabulary size < V:
   a. Count all adjacent symbol pairs in the corpus
   b. Find the most frequent pair (a, b)
   c. Merge all occurrences of (a, b) → ab
   d. Add ab to the vocabulary
3. Return the ordered list of merge rules
```

**Step-by-step example.** Consider the corpus: `"low low low low lowest lowest newer newer wider wider"`

Initial vocabulary (with word frequencies):
```
l o w  (freq: 4)
l o w e s t  (freq: 2)
n e w e r  (freq: 2)
w i d e r  (freq: 2)
```

Iteration 1: Most frequent pair is `(l, o)` (appears 6 times). Merge: `lo`.
Iteration 2: Most frequent pair is `(lo, w)` (appears 6 times). Merge: `low`.
Iteration 3: Most frequent pair is `(e, r)` (appears 4 times). Merge: `er`.
Iteration 4: Most frequent pair is `(e, s)` (appears 2 times). Merge: `es`.
...and so on.

```python
from collections import Counter, defaultdict

def learn_bpe(corpus, num_merges):
    """Learn BPE merge rules from a corpus."""
    # Initialize: split each word into characters
    # Format: list of (token_sequence, frequency) pairs
    vocab = {}
    for word, freq in corpus.items():
        symbols = list(word) + ['</w>']
        vocab[tuple(symbols)] = freq

    merges = []

    for i in range(num_merges):
        # Count all adjacent pairs
        pairs = Counter()
        for symbols, freq in vocab.items():
            for j in range(len(symbols) - 1):
                pairs[(symbols[j], symbols[j+1])] += freq

        if not pairs:
            break

        # Find the most frequent pair
        best_pair = pairs.most_common(1)[0][0]
        merges.append(best_pair)

        # Merge all occurrences of the best pair
        new_vocab = {}
        for symbols, freq in vocab.items():
            new_symbols = []
            j = 0
            while j < len(symbols):
                if (j < len(symbols) - 1 and
                    symbols[j] == best_pair[0] and
                    symbols[j+1] == best_pair[1]):
                    new_symbols.append(best_pair[0] + best_pair[1])
                    j += 2
                else:
                    new_symbols.append(symbols[j])
                    j += 1
            new_vocab[tuple(new_symbols)] = freq
        vocab = new_vocab

        print(f"Merge {i+1}: {best_pair[0]} + {best_pair[1]} "
              f"-> {best_pair[0] + best_pair[1]}")

    return merges
```

### 10.5.2 WordPiece

WordPiece (Schuster & Nakajima, 2012), used by BERT, is similar to BPE but uses a different merge criterion. Instead of merging the most frequent pair, WordPiece merges the pair that maximizes the likelihood of the training data:

$$\text{score}(a, b) = \frac{\text{freq}(ab)}{\text{freq}(a) \times \text{freq}(b)}$$

This favors merging pairs where the combined token is much more frequent than expected by chance, capturing genuine linguistic units rather than mere frequency artifacts.

### 10.5.3 Unigram

The Unigram model (Kudo, 2018), used in SentencePiece, takes the opposite approach to BPE. It starts with a large vocabulary and iteratively removes tokens that least affect the corpus likelihood:

1. Start with a large seed vocabulary (e.g., all substrings up to length $k$).
2. Compute the unigram language model probability of each token.
3. For each token, compute the loss in corpus likelihood if it were removed.
4. Remove the tokens with the smallest loss (e.g., bottom 20%).
5. Repeat until the desired vocabulary size is reached.

The advantage of Unigram is that it provides **multiple segmentations** with associated probabilities, which can be used for regularization during training (subword regularization).

### 10.5.4 SentencePiece and tiktoken

**SentencePiece** (Kudo & Richardson, 2018) is a language-independent tokenizer library that:
- Treats the input as a raw stream of Unicode characters (no pre-tokenization by spaces).
- Supports both BPE and Unigram algorithms.
- Handles languages without explicit word boundaries (Chinese, Japanese, Thai).
- Is used by LLaMA, T5, and many multilingual models.

```python
import sentencepiece as spm

# Train a SentencePiece model
spm.SentencePieceTrainer.train(
    input='training_text.txt',
    model_prefix='tokenizer',
    vocab_size=32000,
    model_type='bpe',            # or 'unigram'
    character_coverage=0.9995,
    byte_fallback=True,          # Handle unknown characters via UTF-8 bytes
)

# Load and use
sp = spm.SentencePieceProcessor()
sp.load('tokenizer.model')
tokens = sp.encode('Hello, world!', out_type=str)
# ['_Hello', ',', '_world', '!']
```

**tiktoken** is OpenAI's tokenizer library, using byte-level BPE:
- Operates directly on UTF-8 byte sequences.
- Pre-tokenizes using regex patterns to split on word boundaries, numbers, and whitespace.
- Used by GPT-3.5, GPT-4, and Claude.

```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
tokens = enc.encode("Hello, world!")
# [9906, 11, 1917, 0]
print(enc.decode(tokens))
# "Hello, world!"
```

### 10.5.5 Vocabulary Size Tradeoffs

| Vocabulary Size | Advantages | Disadvantages |
|---|---|---|
| Small (8K--16K) | Compact embedding table, good for low-resource languages | Longer sequences, slower inference |
| Medium (32K--64K) | Good balance for most use cases | Standard choice |
| Large (100K--256K) | Shorter sequences, better handling of rare words and multilingual text | Large embedding table, sparser training signal per token |

The dominant trend has been toward larger vocabularies: GPT-2 used 50K, LLaMA used 32K, LLaMA 2 used 32K, and many recent models use 100K--256K tokens. Larger vocabularies reduce sequence length (and thus attention computation) at the cost of a larger embedding matrix.

### 10.5.6 Byte-Level BPE

Byte-level BPE (Radford et al., 2019) solves the open-vocabulary problem by operating on raw bytes rather than Unicode characters:

- Every possible input can be tokenized (no "unknown token" needed).
- The base vocabulary is exactly 256 byte values.
- BPE merges are learned on top of these bytes.

This approach is used by GPT-2, GPT-3, GPT-4, and most modern English-centric models. For multilingual models, SentencePiece with byte fallback achieves similar coverage.

---

## 10.6 Training Objectives

The choice of training objective determines what the model learns and how it can be used.

### 10.6.1 Causal Language Modeling (GPT-Style)

The causal (autoregressive, left-to-right) language modeling objective:

$$\mathcal{L}_{\text{CLM}} = -\sum_{t=1}^{T} \log P_\theta(x_t | x_1, \ldots, x_{t-1})$$

The model attends only to previous tokens (using a causal attention mask). This is the objective used by GPT (Radford et al., 2018), GPT-2, GPT-3 (Brown et al., 2020), LLaMA (Touvron et al., 2023), and most modern LLMs.

**Advantages:**
- Natural fit for text generation.
- Simple and scalable.
- Enables in-context learning.

**Disadvantages:**
- Unidirectional: cannot attend to future tokens, which limits performance on some NLU tasks.
- Each token sees only left context during training.

### 10.6.2 Masked Language Modeling (BERT-Style)

The masked language modeling (MLM) objective (Devlin et al., 2019):

1. Randomly select 15% of tokens.
2. Of the selected tokens: 80% are replaced with [MASK], 10% are replaced with a random token, 10% are kept unchanged.
3. The model predicts the original tokens.

$$\mathcal{L}_{\text{MLM}} = -\sum_{t \in \mathcal{M}} \log P_\theta(x_t | x_{\backslash \mathcal{M}})$$

where $\mathcal{M}$ is the set of masked positions.

**Advantages:**
- Bidirectional: each prediction uses both left and right context.
- Strong performance on NLU tasks (classification, NER, QA).

**Disadvantages:**
- Not naturally suited for generation (the model is not trained to produce text left-to-right).
- The [MASK] token creates a train-test mismatch.
- Only 15% of tokens provide a training signal per example.

### 10.6.3 Prefix Language Modeling (T5-Style)

T5 (Raffel et al., 2020) uses a text-to-text framework where all tasks are cast as input-output pairs. The training objective is span corruption:

1. Randomly select spans of tokens (average length 3).
2. Replace each span with a sentinel token (e.g., `<extra_id_0>`).
3. The model generates the corrupted spans.

$$\text{Input: } \text{"The } \langle X \rangle \text{ brown fox } \langle Y \rangle \text{ over the lazy dog"}$$
$$\text{Target: } \langle X \rangle \text{ quick } \langle Y \rangle \text{ jumps}$$

The encoder processes the corrupted input bidirectionally, and the decoder generates the target autoregressively. This combines the bidirectional understanding of MLM with the generative capability of CLM.

### 10.6.4 Choosing an Objective

The field has largely converged on **causal language modeling** for general-purpose models. Key reasons:

1. **Scalability**: CLM scales better with model size and data (Chapter 11).
2. **Generality**: CLM enables both understanding and generation through in-context learning.
3. **Simplicity**: No masking strategy, no encoder-decoder split.
4. **Emergent capabilities**: Few-shot learning and chain-of-thought reasoning emerge most clearly in autoregressive models.

BERT-style models remain relevant for specialized NLU applications where bidirectional context is critical and generation is not needed.

---

## 10.7 Training Stability

Training a large language model is a delicate operation. Models train for weeks or months on thousands of GPUs, and instabilities can waste enormous resources.

### 10.7.1 Loss Spikes

Loss spikes are sudden increases in the training loss, sometimes by orders of magnitude. Common causes:

**Bad data batches.** A batch containing garbled text, extremely long sequences, or adversarial content can produce anomalous gradients. Mitigation: aggressive data filtering (Section 10.4) and gradient clipping.

**Learning rate too high.** If the learning rate exceeds the stability threshold for the current loss landscape, parameters can overshoot. Mitigation: warmup (Section 10.7.3) and careful LR scheduling.

**Gradient explosion.** Gradients can grow exponentially through deep networks, particularly during early training. Mitigation: gradient clipping to a maximum norm (typically 1.0).

**Numerical overflow.** In FP16 training, intermediate values can exceed the representable range. Mitigation: use BF16 or carefully tuned loss scaling.

**Recovery strategies:**
1. **Skip the bad batch**: If the spike is isolated, simply discarding the offending batch and continuing from the last good state can work.
2. **Roll back**: Restore from a recent checkpoint (before the spike) and resume with a lower learning rate.
3. **Reduce learning rate**: Permanently reduce the learning rate if spikes recur.

### 10.7.2 Z-Loss Regularization

The PaLM paper (Chowdhery et al., 2022) introduced z-loss regularization to stabilize training. The logits (pre-softmax values) in large models can grow unboundedly, leading to numerical instability. Z-loss adds a penalty on the log-sum-exp of logits:

$$\mathcal{L}_z = \alpha_z \cdot \log^2 \left( \sum_j \exp(z_j) \right)$$

where $z_j$ are the logits and $\alpha_z$ is a small coefficient (e.g., $10^{-4}$). This encourages the logits to remain in a numerically stable range without significantly affecting the predicted distribution.

```python
def z_loss(logits, coefficient=1e-4):
    """Z-loss regularization for training stability."""
    log_z = torch.logsumexp(logits, dim=-1)
    return coefficient * (log_z ** 2).mean()
```

### 10.7.3 Warmup Necessity

Learning rate warmup --- gradually increasing the learning rate from zero (or a very small value) to the target value over the first few hundred to few thousand steps --- is essential for stable training. Without warmup:

- Adam's moment estimates are initialized to zero and are poor approximations during early training.
- The loss landscape around the random initialization may have high curvature.
- Large initial updates can push the model into unstable regions.

A typical warmup schedule increases the learning rate linearly over 0.1--2% of total training steps:

$$\text{lr}(t) = \begin{cases}
\text{lr}_{\max} \cdot \frac{t}{T_{\text{warmup}}} & \text{if } t \leq T_{\text{warmup}} \\
\text{lr}_{\max} \cdot \text{decay}(t) & \text{if } t > T_{\text{warmup}}
\end{cases}$$

After warmup, the learning rate typically follows a cosine decay schedule:

$$\text{decay}(t) = \frac{1}{2}\left(1 + \cos\left(\pi \cdot \frac{t - T_{\text{warmup}}}{T_{\text{total}} - T_{\text{warmup}}}\right)\right)$$

---

## 10.8 Data Mixing

The composition of the training data --- the relative proportions of web text, books, code, scientific papers, and other sources --- significantly affects model capabilities.

### 10.8.1 Domain Weighting Strategies

**Uniform mixing**: Sample from each domain proportionally to its size. Simple but suboptimal: web data dominates, and high-quality sources like books and Wikipedia are underrepresented.

**Upsampling high-quality sources**: The LLaMA approach (Touvron et al., 2023) upsamples Wikipedia and books by 2--3x relative to their natural proportion. This improves factual knowledge and writing quality.

**Learned mixing**: The DoReMi method (Xie et al., 2023) learns domain weights by training a small proxy model and adjusting weights to minimize worst-case excess loss across domains. This is a principled approach but adds complexity.

### 10.8.2 Typical Data Mixes

| Source | LLaMA (%) | Chinchilla (%) | GPT-3 (%) |
|---|---|---|---|
| Web (Common Crawl) | 67 | ~80 | 60 |
| Books | 4.5 | ~8 | 16 |
| Code (GitHub) | 4.5 | -- | -- |
| Wikipedia | 4.5 | ~3 | 3 |
| Scientific papers | 2.5 | ~3 | -- |
| StackExchange | 2 | -- | -- |
| Other | 15 | ~6 | 21 |

The trend is toward **more code and more diverse web data**, as code training improves reasoning and diverse web data improves factual coverage.

---

## 10.9 Data Curriculum

Rather than presenting data uniformly throughout training, a **data curriculum** orders or weights data to optimize learning.

### 10.9.1 Ordering by Difficulty

Training on easier examples first (shorter documents, simpler language) and progressively introducing harder examples can improve training stability and final performance. However, defining "difficulty" for language modeling is non-trivial. Proxies include:

- **Document length**: Shorter documents first.
- **Perplexity**: Lower-perplexity (more predictable) documents first.
- **Source quality**: Higher-quality sources first, web data later.

### 10.9.2 Upsampling High-Quality Sources

A pragmatic curriculum strategy is to increase the proportion of high-quality data in later stages of training. For example:

- **Phase 1** (80% of training): Standard data mix with web-heavy composition.
- **Phase 2** (20% of training): Increased proportion of books, Wikipedia, curated web, and code.

This "annealing" approach, used by LLaMA 3 and other recent models, ensures the model's final representations are shaped by high-quality content.

### 10.9.3 Epoch Considerations

**Should data be repeated?** For most large-scale training runs, the answer is: minimally. Muennighoff et al. (2023) showed that:

- Training for 1 epoch on unique data is optimal.
- Beyond 4 epochs, returns diminish sharply, and after ~10 epochs, additional passes can degrade performance.
- If data is limited, it is better to train for fewer steps than to repeat data excessively.

For high-quality subsets (Wikipedia, curated books), 2--4 epochs are acceptable. For web data, a single pass is preferred.

---

## 10.10 Document Packing

Language models process fixed-length sequences, but documents vary in length. Document packing addresses this mismatch.

### 10.10.1 Concatenation with Special Tokens

The standard approach is to concatenate documents end-to-end, separated by a special end-of-document token:

```
[Doc1 tokens] <EOS> [Doc2 tokens] <EOS> [Doc3 tokens] <EOS> ...
```

This concatenated stream is then split into fixed-length chunks for training. The advantages are:

- No padding waste (every token in every sequence contributes to the loss).
- Simple implementation.

### 10.10.2 Attention Masking Across Documents

A subtlety: when documents are concatenated, should tokens in Document 2 attend to tokens in Document 1? There are two approaches:

**Full attention** (the simpler approach): Allow attention across document boundaries. Tokens in Document 2 can attend to Document 1 tokens. This is simpler to implement but introduces a spurious dependency between unrelated documents.

**Document-masked attention**: Apply an attention mask that prevents cross-document attention. Each document can only attend to tokens within its own boundaries:

```python
def create_document_attention_mask(document_ids, seq_len):
    """Create an attention mask that prevents cross-document attention.

    Args:
        document_ids: Tensor of shape (seq_len,) indicating which
                      document each token belongs to.
    Returns:
        mask: Boolean tensor of shape (seq_len, seq_len).
    """
    # Tokens can attend to other tokens in the same document
    mask = document_ids.unsqueeze(0) == document_ids.unsqueeze(1)
    # Also apply causal mask (can't attend to future tokens)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    return mask & causal_mask
```

Document-masked attention is more principled but adds implementation complexity and slightly reduces throughput (the attention computation must handle a non-standard mask pattern). In practice, for very long context windows (8K+), the spurious dependencies from full attention become negligible, and many implementations use full attention for simplicity.

---

## 10.11 Checkpoint Management

Training runs that cost millions of dollars and run for weeks cannot afford to lose progress. Robust checkpoint management is essential.

### 10.11.1 Saving Frequency

Checkpoints should be saved:
- **Regularly**: Every 500--2000 steps, or every 1--4 hours.
- **At milestones**: At the end of each epoch or data phase.
- **Before risky operations**: Before changing hyperparameters or data mix.

Each checkpoint includes: model weights, optimizer states, learning rate scheduler state, data loader state (to resume from the exact position), random number generator states, and training metadata (step, loss, etc.).

For a 7B model with AdamW optimizer, each checkpoint is approximately:
$$7\text{B} \times (2 + 4 + 4 + 4) = 98 \text{ GB}$$

(FP16 weights + FP32 optimizer states). Storage costs add up quickly for large models.

### 10.11.2 Checkpoint Averaging

Averaging the weights of the last $k$ checkpoints can improve stability and generalization:

$$\theta_{\text{avg}} = \frac{1}{k} \sum_{i=1}^{k} \theta_i$$

This is an approximation to Stochastic Weight Averaging (SWA) and smooths out oscillations in the loss landscape. It is particularly useful when training is terminated and one wants the best possible final model.

### 10.11.3 Evaluating Intermediate Checkpoints

Running evaluations at intermediate checkpoints provides crucial diagnostics:

- **Training loss**: Should decrease smoothly. Sudden increases indicate instability.
- **Validation loss**: Should decrease, then plateau. If it increases while training loss decreases, the model is overfitting.
- **Downstream benchmarks**: Run a suite of benchmarks (MMLU, HellaSwag, ARC) every few thousand steps to track capability emergence.
- **Perplexity on held-out domains**: Tracks whether the model is learning each domain.

---

## 10.12 Modern Optimizers

### 10.12.1 AdamW

AdamW (Loshchilov & Hutter, 2019) is the standard optimizer for transformer training. It decouples weight decay from the gradient update:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = m_t / (1 - \beta_1^t)$$
$$\hat{v}_t = v_t / (1 - \beta_2^t)$$
$$\theta_t = \theta_{t-1} - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \right)$$

where $\lambda$ is the weight decay coefficient. The key insight: in standard Adam, weight decay is applied to the gradient-scaled update, which means the effective regularization depends on the adaptive learning rate. AdamW applies weight decay directly to the weights, providing consistent regularization regardless of the gradient magnitude.

Standard hyperparameters for LLM pretraining:
- $\beta_1 = 0.9$, $\beta_2 = 0.95$
- $\epsilon = 10^{-8}$
- Weight decay $\lambda = 0.1$
- Learning rate: $3 \times 10^{-4}$ (varies with model size)

### 10.12.2 Lion

Lion (Chen et al., 2024) is a simpler optimizer that uses only the **sign** of the momentum, not its magnitude:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$\theta_t = \theta_{t-1} - \eta \left( \text{sign}(\beta_2 m_{t-1} + (1 - \beta_2) g_t) + \lambda \theta_{t-1} \right)$$

Lion's advantages:
- **Memory**: Only one momentum buffer (vs. two for Adam), saving ~33% optimizer memory.
- **Simplicity**: The update magnitude is uniform (just the sign), so the learning rate directly controls step size.
- **Effectiveness**: Competitive with or slightly better than AdamW on many benchmarks.

Lion typically requires a lower learning rate than AdamW (roughly 3--10x lower) and higher weight decay.

### 10.12.3 Muon

Muon (Jordan et al., 2024) is a newer optimizer based on the principle of steepest descent in the spectral norm. For weight matrices, it computes the update direction that has the largest impact on the output, as measured by the matrix spectral norm. Muon applies Newton-Schulz iterations to approximate the matrix square root of the gradient outer product, yielding an update that is closer to natural gradient descent.

Muon has shown promising results on small to medium-scale training runs, often reaching the same loss as AdamW in fewer steps. However, its computational overhead (matrix operations per step) and limited large-scale validation mean it is not yet the standard choice for production training.

---

## 10.13 Training a Small Language Model from Scratch

Let us now synthesize everything in this chapter with a complete training pipeline. We will train a small GPT-style model on a text dataset, following the approach of Karpathy's nanoGPT.

### 10.13.1 Data Preparation

```python
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

# Load a text dataset
dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

# Use a pretrained tokenizer (or train your own with SentencePiece)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

def tokenize_function(examples):
    return tokenizer(examples["text"], return_attention_mask=False)

tokenized = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
)

# Concatenate all tokens and split into fixed-length chunks
block_size = 1024

def group_texts(examples):
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated["input_ids"])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i:i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_dataset = tokenized.map(group_texts, batched=True)
```

### 10.13.2 Model Definition

```python
import torch.nn as nn
import math

class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Use PyTorch's scaled_dot_product_attention (Flash Attention)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.attn_dropout.p if self.training else 0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)
        # Apply special scaled init to residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0,
                                std=0.02 / math.sqrt(2 * config.n_layer))

        n_params = sum(p.numel() for p in self.parameters())
        print(f"Number of parameters: {n_params / 1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss
```

### 10.13.3 Training Loop

```python
from torch.utils.data import DataLoader

# Initialize
config = GPTConfig()
model = GPT(config).cuda()

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1,
)

# Learning rate scheduler with warmup and cosine decay
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

warmup_steps = 2000
total_steps = 100000

warmup_scheduler = LinearLR(
    optimizer, start_factor=1e-8 / 3e-4, total_iters=warmup_steps
)
cosine_scheduler = CosineAnnealingLR(
    optimizer, T_max=total_steps - warmup_steps, eta_min=3e-5
)
scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[warmup_steps],
)

# Training loop
model.train()
dataloader = DataLoader(lm_dataset["train"], batch_size=8, shuffle=True)

for step, batch in enumerate(dataloader):
    if step >= total_steps:
        break

    input_ids = torch.tensor(batch["input_ids"]).cuda()
    targets = torch.tensor(batch["labels"]).cuda()

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits, loss = model(input_ids, targets)

    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)

    if step % 100 == 0:
        print(f"Step {step}: loss = {loss.item():.4f}, "
              f"lr = {scheduler.get_last_lr()[0]:.2e}")

    # Save checkpoint
    if step % 5000 == 0 and step > 0:
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss.item(),
        }, f'checkpoint_step_{step}.pt')
```

### 10.13.4 Evaluation

```python
@torch.no_grad()
def evaluate(model, eval_dataloader, max_batches=100):
    model.eval()
    total_loss = 0
    total_tokens = 0

    for i, batch in enumerate(eval_dataloader):
        if i >= max_batches:
            break
        input_ids = torch.tensor(batch["input_ids"]).cuda()
        targets = torch.tensor(batch["labels"]).cuda()

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            _, loss = model(input_ids, targets)

        total_loss += loss.item() * input_ids.numel()
        total_tokens += input_ids.numel()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    model.train()
    return avg_loss, perplexity

# Run evaluation
eval_dataloader = DataLoader(lm_dataset["validation"], batch_size=8)
val_loss, val_ppl = evaluate(model, eval_dataloader)
print(f"Validation loss: {val_loss:.4f}, Perplexity: {val_ppl:.2f}")
```

### 10.13.5 Generation

```python
@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=100,
             temperature=0.8, top_k=40):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').cuda()

    for _ in range(max_new_tokens):
        # Crop to block_size if needed
        idx_cond = input_ids if input_ids.size(1) <= model.config.block_size \
                   else input_ids[:, -model.config.block_size:]

        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        # Top-k filtering
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')

        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(input_ids[0])

# Generate text
print(generate(model, tokenizer, "The future of artificial intelligence"))
```

---

## 10.14 Summary

Pretraining a large language model is an end-to-end engineering challenge that spans data engineering, numerical computing, optimization theory, and systems design. The key lessons from this chapter:

1. **Data is paramount**: The quality and composition of the training data determines the model's capabilities more than any architectural choice.
2. **Deduplication matters**: Removing duplicate and near-duplicate content improves both training efficiency and final model quality.
3. **Tokenization is a design decision**: Vocabulary size, algorithm choice, and byte-level vs. character-level all affect the model's effective context window and multilingual capabilities.
4. **Stability requires vigilance**: Loss spikes, gradient explosions, and numerical overflow are ever-present risks that require monitoring, prevention, and recovery strategies.
5. **Data mixing is an art**: The relative proportions of web, books, code, and other sources must be carefully tuned, and the optimal mix likely varies across training stages.

---

## Exercises

1. **BPE From Scratch**: Implement the BPE algorithm from scratch (without using any tokenizer library). Train it on a small corpus and compare the resulting vocabulary to that produced by `sentencepiece` with the same vocabulary size. How similar are the merge rules?

2. **Deduplication Impact**: Take a text dataset and create three versions: (a) no deduplication, (b) exact deduplication only, (c) exact + MinHash near-deduplication. Train a small language model on each version for the same number of steps. Compare validation loss and generation quality.

3. **Quality Filtering Pipeline**: Implement the full quality filtering pipeline from Section 10.4 (heuristic filters + perplexity filtering). Apply it to a Common Crawl sample and measure the fraction of data removed at each stage. Manually inspect 100 removed and 100 kept documents --- do the filters make reasonable decisions?

4. **Training Objective Comparison**: Train three small models with the same architecture but different objectives: (a) causal LM, (b) masked LM, (c) prefix LM. Compare them on text generation, text classification, and question answering. Which objective excels at which task?

5. **Loss Spike Investigation**: Deliberately introduce training instabilities by (a) using an excessively high learning rate, (b) removing warmup, (c) injecting corrupted data batches. Document the resulting loss curves and practice recovery strategies (rollback to checkpoint, LR reduction).

6. **Optimizer Comparison**: Train the same model with AdamW and Lion. Compare convergence speed, final loss, memory usage, and wall-clock time. Does Lion's memory advantage translate to practical benefits (e.g., larger batch size)?

7. **Data Curriculum Experiment**: Train two identical models: one with uniform random data sampling, and one with a curriculum (easier data first, harder data later). Define "difficulty" using document perplexity under a reference model. Does the curriculum improve final performance?

---

## References

- Brown, T. B., et al. (2020). Language Models are Few-Shot Learners. *Advances in Neural Information Processing Systems, 33*, 1877--1901.
- Chen, X., Liang, C., Huang, D., Real, E., Wang, K., Liu, H., Pham, H., Dong, X., Luber, T., Cho, K., Hsieh, C., & Le, Q. V. (2024). Symbolic Discovery of Optimization Algorithms. *Advances in Neural Information Processing Systems, 36*.
- Chowdhery, A., et al. (2022). PaLM: Scaling Language Modeling with Pathways. *arXiv preprint arXiv:2204.02311*.
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *Proceedings of NAACL-HLT 2019*, 4171--4186.
- Gao, L., et al. (2020). The Pile: An 800GB Dataset of Diverse Text for Language Modeling. *arXiv preprint arXiv:2101.00027*.
- Jordan, K., et al. (2024). Muon: An Optimizer for Hidden Layers. *arXiv preprint*.
- Kocetkov, D., et al. (2022). The Stack: 3 TB of Permissively Licensed Source Code. *arXiv preprint arXiv:2211.15533*.
- Kudo, T. (2018). Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates. *Proceedings of ACL 2018*, 66--75.
- Kudo, T., & Richardson, J. (2018). SentencePiece: A Simple and Language Independent Subword Tokenizer and Detokenizer for Neural Text Processing. *Proceedings of EMNLP 2018 (System Demonstrations)*, 66--71.
- Lee, K., Ippolito, D., Nushi, A., Choi, Y., & Shin, R. (2022). Deduplicating Training Data Makes Language Models Better. *Proceedings of ACL 2022*, 8424--8445.
- Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. *Proceedings of ICLR 2019*.
- Madaan, A., et al. (2022). Language Models of Code are Few-Shot Commonsense Learners. *Proceedings of EMNLP 2022*, 1384--1403.
- Muennighoff, N., et al. (2023). Scaling Data-Constrained Language Models. *Advances in Neural Information Processing Systems, 36*.
- Penedo, G., et al. (2023). The RefinedWeb Dataset for Falcon LLM: Outperforming Curated Corpora with Web Data, and Web Data Only. *arXiv preprint arXiv:2306.01116*.
- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language Models are Unsupervised Multitask Learners. *OpenAI Blog*.
- Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018). Improving Language Understanding by Generative Pre-Training. *OpenAI Blog*.
- Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., & Liu, P. J. (2020). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. *Journal of Machine Learning Research, 21*(140), 1--67.
- Schuster, M., & Nakajima, K. (2012). Japanese and Korean Voice Search. *Proceedings of ICASSP 2012*, 5149--5152.
- Sennrich, R., Haddow, B., & Birch, A. (2016). Neural Machine Translation of Rare Words with Subword Units. *Proceedings of ACL 2016*, 1715--1725.
- Soldaini, L., et al. (2024). Dolma: An Open Corpus of Three Trillion Tokens for Language Model Pretraining Research. *arXiv preprint arXiv:2402.00159*.
- Touvron, H., et al. (2023). LLaMA: Open and Efficient Foundation Language Models. *arXiv preprint arXiv:2302.13971*.
- Wei, J., et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. *Advances in Neural Information Processing Systems, 35*.
- Wenzek, G., et al. (2020). CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data. *Proceedings of LREC 2020*, 4003--4012.
- Xie, S. M., et al. (2023). DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining. *Advances in Neural Information Processing Systems, 36*.
