# Chapter 15: RAG and Retrieval Systems

## Learning Objectives

By the end of this chapter, you will be able to:

1. **Explain the RAG paradigm** — why retrieval-augmented generation was invented, how it addresses hallucination, knowledge cutoffs, and domain specificity, and how it compares to fine-tuning.
2. **Design a document processing pipeline** — parse diverse formats (PDF, HTML, tables), select appropriate chunking strategies, and extract metadata for downstream retrieval.
3. **Select and deploy embedding models** — trace the evolution from Word2Vec through modern instruction-tuned embeddings, understand contrastive training objectives, and make informed tradeoffs between dimensionality and quality.
4. **Build and query vector databases** — implement ANN indices with FAISS, deploy managed solutions like Pinecone and Weaviate, and reason about index type tradeoffs.
5. **Implement dense, sparse, and hybrid retrieval** — build bi-encoder pipelines, configure BM25, and fuse results using Reciprocal Rank Fusion.
6. **Apply advanced retrieval techniques** — deploy cross-encoder rerankers, ColBERT late interaction, HyDE, GraphRAG, and multimodal retrieval.
7. **Evaluate RAG systems rigorously** — use the RAGAS framework to measure context precision, context recall, faithfulness, and answer relevance.
8. **Architect production RAG systems** — design for caching, routing, monitoring, and cost efficiency at scale.

---

## 15.1 The RAG Paradigm

### 15.1.1 Motivation

Large language models are remarkable tools for generating fluent, coherent text. Yet they suffer from three fundamental limitations that make them unreliable as standalone knowledge systems.

**Hallucination.** LLMs generate text by sampling from learned probability distributions over tokens. When a model encounters a query outside or at the boundary of its training distribution, it does not abstain — it confabulates. The output reads convincingly but contains factual errors. In domains such as medicine, law, and finance, hallucinated content can have severe consequences (Ji et al., 2023).

**Knowledge cutoffs.** A model's parametric knowledge is frozen at the time of its last training run. GPT-4, for instance, was trained on data through April 2023; any event, publication, or regulatory change after that date is invisible to the model. For organizations whose competitive advantage depends on current information — news agencies, trading desks, customer support teams — this staleness is unacceptable.

**Domain specificity.** Even within their training window, LLMs allocate capacity according to the frequency distribution of the training corpus. Niche domains — internal company documentation, proprietary research, rare languages — receive minimal representation. Fine-tuning can help but is expensive, slow to iterate, and risks catastrophic forgetting of general capabilities.

Retrieval-Augmented Generation (RAG) addresses all three limitations by decoupling the knowledge source from the reasoning engine. Instead of relying solely on parametric memory, a RAG system retrieves relevant documents from an external corpus and presents them to the LLM as context for generation (Lewis et al., 2020).

### 15.1.2 Architecture Overview

The canonical RAG pipeline follows three stages:

1. **Retrieve.** Given a user query $q$, search an external knowledge base $\mathcal{D}$ for the top-$k$ most relevant documents (or document chunks) $\{d_1, d_2, \ldots, d_k\}$.
2. **Augment.** Construct an augmented prompt by concatenating the retrieved documents with the original query: $\text{prompt} = \text{SystemInstructions} \oplus d_1 \oplus d_2 \oplus \cdots \oplus d_k \oplus q$.
3. **Generate.** Feed the augmented prompt to an LLM to produce the final response $a = \text{LLM}(\text{prompt})$.

This architecture is deceptively simple. The complexity — and the subject of this chapter — lies in the dozens of design decisions within each stage: how to chunk documents, which embedding model to use, which index structure to deploy, how to rerank results, and how to evaluate the end-to-end system.

### 15.1.3 RAG vs. Fine-Tuning

| Dimension | RAG | Fine-Tuning |
|-----------|-----|-------------|
| Knowledge freshness | Updated by re-indexing documents | Frozen at training time |
| Setup cost | Moderate (embedding + indexing) | High (GPU hours, data curation) |
| Latency | Higher (retrieval adds ~100–500ms) | Lower (single forward pass) |
| Hallucination control | Grounded in retrieved text | Still possible |
| Domain adaptation | Add domain documents | Train on domain data |
| Auditability | Can cite source documents | Black-box parametric knowledge |
| Scalability of knowledge | Millions of documents easily | Limited by model capacity |

In practice, the two approaches are complementary. Fine-tuning teaches a model *how* to reason and respond in a particular style or domain; RAG provides the *what* — the specific facts and evidence the model should use. Many production systems combine both: a fine-tuned model serves as the generator, and RAG supplies grounding context.

---

## 15.2 Document Processing

### 15.2.1 Parsing

Real-world knowledge bases are heterogeneous. A single organization may store information across PDFs, HTML pages, Markdown files, Word documents, spreadsheets, and databases. The first step in any RAG pipeline is normalizing these diverse formats into clean text.

**PDF parsing** is notoriously difficult. PDFs are a page-description language, not a semantic format; they specify where to draw glyphs on a page, not what those glyphs mean. Tools include:

- **PyPDF2 / PyMuPDF (fitz):** Extract raw text layer. Fast but loses structure (headers, tables, columns).
- **pdfplumber:** Better at table extraction by analyzing character positions.
- **Unstructured.io:** Open-source library that combines multiple parsing strategies, including OCR for scanned documents.
- **Amazon Textract / Google Document AI:** Cloud-based OCR with layout understanding.

```python
# Parsing a PDF with PyMuPDF
import fitz  # PyMuPDF

def parse_pdf(path: str) -> list[dict]:
    """Extract text from each page of a PDF."""
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        pages.append({
            "page_number": i + 1,
            "text": text.strip(),
            "metadata": {"source": path, "page": i + 1}
        })
    return pages
```

**HTML parsing** is more structured but requires stripping boilerplate (navigation bars, footers, ads). Libraries like `BeautifulSoup` and `trafilatura` extract main content. For web-scale ingestion, tools like `Crawl4AI` combine crawling with intelligent content extraction.

**Table extraction** deserves special attention because tables encode relational information that is destroyed by naive text extraction. Strategies include converting tables to Markdown format, serializing rows as natural language sentences, or preserving them as structured metadata.

### 15.2.2 Chunking Strategies

LLMs have finite context windows, and embedding models have even shorter maximum input lengths (typically 512 tokens). Documents must be split into chunks that are small enough to embed but large enough to preserve semantic coherence.

**Fixed-size chunking** splits text into chunks of $n$ characters or $n$ tokens, with an overlap of $m$:

```python
def fixed_size_chunk(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Split text into fixed-size chunks with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks
```

This is simple and predictable but frequently splits sentences and paragraphs mid-thought.

**Recursive character splitting** (popularized by LangChain) attempts to split on natural boundaries in a hierarchical order: first on double newlines (paragraph breaks), then single newlines, then sentences, then words. This preserves semantic units better than fixed-size splitting.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = splitter.split_text(document_text)
```

**Semantic chunking** uses an embedding model to detect topic shifts. It computes embeddings for sentences or small segments, then splits where the cosine similarity between consecutive segments drops below a threshold:

```python
import numpy as np
from sentence_transformers import SentenceTransformer

def semantic_chunk(sentences: list[str], model_name: str = "all-MiniLM-L6-v2",
                   threshold: float = 0.75) -> list[list[str]]:
    """Group sentences into semantic chunks based on embedding similarity."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences)

    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        similarity = np.dot(embeddings[i], embeddings[i-1]) / (
            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i-1])
        )
        if similarity < threshold:
            chunks.append(current_chunk)
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])

    chunks.append(current_chunk)
    return chunks
```

**Document-structure-aware chunking** leverages the hierarchical structure of documents — chapters, sections, subsections — to create chunks that respect logical boundaries. This requires parsing document structure first (e.g., using heading levels in Markdown or HTML) and then chunking within those structural units.

**Chunk overlap** is a critical parameter. Too little overlap and context is lost at chunk boundaries; too much and the index becomes bloated with redundant content. A common heuristic is 10–20% overlap relative to chunk size.

### 15.2.3 Metadata Extraction

Each chunk should carry metadata that enables filtering, attribution, and context reconstruction:

- **Source document** (file path, URL, document ID)
- **Position** (page number, section heading, chunk index)
- **Temporal information** (publication date, last modified)
- **Domain tags** (department, product, topic)
- **Relationships** (parent document, sibling chunks, linked entities)

Metadata enables hybrid retrieval strategies (e.g., "retrieve only from documents published after 2024") and is essential for citation in the generated response.

---

## 15.3 Embedding Models

### 15.3.1 Evolution of Text Embeddings

The history of text embeddings traces a path from word-level representations to instruction-tuned sentence embeddings optimized for retrieval.

**Word2Vec** (Mikolov et al., 2013) learned dense word vectors by predicting context words (Skip-gram) or predicting a word from its context (CBOW). Word2Vec demonstrated that arithmetic relationships hold in embedding space ($\vec{\text{king}} - \vec{\text{man}} + \vec{\text{woman}} \approx \vec{\text{queen}}$), but it produced a single vector per word regardless of context.

**GloVe** (Pennington et al., 2014) combined the advantages of global matrix factorization (like LSA) with local context window methods (like Word2Vec). It explicitly factorized the word co-occurrence matrix, producing embeddings where dot products approximate log co-occurrence probabilities.

**Sentence-BERT** (Reimers & Gurevych, 2019) applied siamese and triplet network structures to pretrained BERT to derive fixed-size sentence embeddings. This was the first practical approach for computing semantically meaningful sentence-level embeddings suitable for similarity search.

**OpenAI text-embedding-3-small/large** (2024) introduced Matryoshka Representation Learning, allowing embeddings to be truncated to smaller dimensions without retraining. The `text-embedding-3-small` model produces 1536-dimensional vectors that can be truncated to 512 or 256 dimensions with modest quality degradation.

**BGE (BAAI General Embedding)** models from the Beijing Academy of Artificial Intelligence achieved state-of-the-art performance on the MTEB benchmark. The `bge-large-en-v1.5` model uses instruction-prefixed queries ("Represent this sentence for searching relevant passages:") to specialize embeddings for retrieval.

**E5** (Wang et al., 2022) trained embeddings using a contrastive objective on large-scale text pairs, with the key innovation of using prompted prefixes ("query:" and "passage:") to differentiate query and document embeddings.

### 15.3.2 Contrastive Learning for Embeddings

Modern embedding models are trained with contrastive objectives. The dominant loss function is **InfoNCE** (Oord et al., 2018), which encourages the model to place positive (relevant) pairs close together in embedding space while pushing negative (irrelevant) pairs apart.

Given a query $q$ and its positive document $d^+$, with $N-1$ negative documents $\{d_1^-, d_2^-, \ldots, d_{N-1}^-\}$, the InfoNCE loss is:

$$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(q, d^+) / \tau)}{\exp(\text{sim}(q, d^+) / \tau) + \sum_{i=1}^{N-1} \exp(\text{sim}(q, d_i^-) / \tau)}$$

where $\text{sim}(\cdot, \cdot)$ is cosine similarity and $\tau$ is a temperature parameter controlling the sharpness of the distribution. In-batch negatives — treating all other documents in a training batch as negatives — provide a computationally efficient source of hard negatives.

### 15.3.3 Embedding Dimensions and Quality Tradeoffs

Higher-dimensional embeddings capture more nuance but incur costs:

| Dimension | Storage per 1M docs | Index build time | Retrieval latency | Quality (MTEB) |
|-----------|---------------------|-------------------|-------------------|-----------------|
| 256 | ~1 GB | Fast | Low | Good |
| 768 | ~3 GB | Moderate | Moderate | Very Good |
| 1536 | ~6 GB | Slow | Higher | Excellent |
| 3072 | ~12 GB | Very Slow | Highest | Marginal gain |

Matryoshka Representation Learning (Kusupati et al., 2022) offers a principled way to train embeddings that remain useful at multiple truncation levels, enabling practitioners to choose their quality-cost tradeoff at deployment time without retraining.

---

## 15.4 Vector Databases

### 15.4.1 The Need for Approximate Nearest Neighbors

Given a query embedding $\mathbf{q} \in \mathbb{R}^d$ and a corpus of $N$ document embeddings $\{\mathbf{d}_1, \ldots, \mathbf{d}_N\}$, exact nearest neighbor search requires computing $N$ distance calculations — $O(Nd)$ time. For $N = 10^8$ documents with $d = 768$, this is prohibitively expensive. Approximate Nearest Neighbor (ANN) algorithms trade a small amount of recall for dramatic speedups.

### 15.4.2 FAISS

Facebook AI Similarity Search (Johnson et al., 2019) is the foundational library for ANN search. It provides several index types:

**IVF (Inverted File Index).** Partitions the embedding space into $C$ Voronoi cells using $k$-means clustering. At query time, only the $n_{\text{probe}}$ nearest clusters are searched, reducing the search space by a factor of $C / n_{\text{probe}}$. The tradeoff between speed and recall is controlled by $n_{\text{probe}}$: higher values improve recall but increase latency.

**HNSW (Hierarchical Navigable Small World).** Builds a multi-layer graph where each layer is a navigable small-world graph with decreasing density. Search begins at the top (sparsest) layer and greedily descends to the bottom (densest) layer, navigating toward the nearest neighbors. HNSW provides excellent recall-speed tradeoffs but requires more memory than IVF.

**PQ (Product Quantization).** Compresses $d$-dimensional vectors by splitting each vector into $m$ sub-vectors, quantizing each sub-vector independently using a $k$-means codebook. A 768-dimensional float32 vector (3072 bytes) can be compressed to as few as 96 bytes ($m = 96$ sub-vectors, each encoded with 1 byte). PQ is often combined with IVF (IVF-PQ) for compressed, partitioned search.

```python
import faiss
import numpy as np

d = 768          # Embedding dimension
n = 1_000_000   # Number of documents
nlist = 1024     # Number of IVF clusters
m = 48           # Number of PQ sub-vectors

# Generate synthetic embeddings
xb = np.random.random((n, d)).astype('float32')
faiss.normalize_L2(xb)

# Build IVF-PQ index
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)  # 8 bits per sub-vector
index.train(xb)
index.add(xb)

# Search
xq = np.random.random((1, d)).astype('float32')
faiss.normalize_L2(xq)
index.nprobe = 10  # Search 10 nearest clusters
distances, indices = index.search(xq, k=10)
```

### 15.4.3 Managed Vector Databases

**Pinecone** is a fully managed vector database offering serverless and pod-based deployments. It handles index management, scaling, and backups, making it attractive for teams that want to avoid infrastructure overhead. Pinecone supports metadata filtering and sparse-dense hybrid search natively.

**ChromaDB** is a lightweight, open-source embedding database designed for prototyping and small-to-medium workloads. It runs in-process or as a client-server, integrates natively with LangChain and LlamaIndex, and stores data persistently with SQLite.

```python
import chromadb
from chromadb.utils import embedding_functions

# Initialize ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Create collection
collection = client.get_or_create_collection(
    name="documents",
    embedding_function=ef,
    metadata={"hnsw:space": "cosine"}
)

# Add documents
collection.add(
    documents=["RAG improves LLM accuracy", "Vector databases enable similarity search"],
    metadatas=[{"source": "paper1"}, {"source": "paper2"}],
    ids=["doc1", "doc2"]
)

# Query
results = collection.query(
    query_texts=["How does retrieval help language models?"],
    n_results=5
)
```

**Weaviate** distinguishes itself with native hybrid search (combining vector and keyword search in a single query), a GraphQL API, and modular vectorization (plug in any embedding model). It supports multi-tenancy and is a strong choice for production deployments requiring both semantic and keyword search.

**Milvus** is an open-source vector database designed for enterprise-scale workloads. Built on a microservices architecture, it supports billions of vectors, GPU-accelerated indexing, and multiple index types (IVF, HNSW, DiskANN). Zilliz Cloud provides a managed version.

**Qdrant** offers a Rust-based vector database with strong filtering capabilities (payload-based filtering during search), quantization support, and a gRPC API for low-latency applications.

### 15.4.4 Index Types and Tradeoffs

| Index Type | Build Time | Query Latency | Memory | Recall@10 | Best For |
|-----------|-----------|---------------|---------|-----------|----------|
| Flat (Exact) | O(1) | O(Nd) | O(Nd) | 100% | Small corpora (<100K) |
| IVF-Flat | O(Nd) | O(n_probe × N/C × d) | O(Nd) | 95–99% | Medium corpora |
| IVF-PQ | O(Nd) | O(n_probe × N/C) | O(Nm/8) | 85–95% | Large corpora, memory-constrained |
| HNSW | O(N log N) | O(log N) | O(Nd + edges) | 97–99% | Low-latency requirements |
| DiskANN | O(N log N) | O(log N) disk | O(N) RAM + disk | 95–99% | Very large corpora, SSD-based |

---

## 15.5 Dense Retrieval

### 15.5.1 Bi-Encoder Architecture

Dense retrieval uses learned embeddings to represent both queries and documents as dense vectors. The **bi-encoder** architecture (also called dual-encoder) encodes queries and documents independently using separate (or shared) encoder networks:

$$\mathbf{q} = E_q(\text{query}), \quad \mathbf{d} = E_d(\text{document})$$

Relevance is scored as the similarity between the two vectors:

$$\text{score}(q, d) = \text{sim}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\| \|\mathbf{d}\|}$$

The critical advantage of bi-encoders is that document embeddings can be precomputed and indexed offline. At query time, only the query needs to be encoded, and retrieval reduces to an ANN search — typically sub-millisecond for millions of documents.

### 15.5.2 DPR (Dense Passage Retrieval)

Karpukhin et al. (2020) introduced DPR, a pioneering dense retrieval system that fine-tuned two independent BERT encoders (one for queries, one for passages) using contrastive learning. DPR demonstrated that dense retrieval could outperform BM25 on open-domain question answering benchmarks, particularly for questions requiring semantic understanding rather than keyword matching.

The DPR training objective is a variant of InfoNCE. Given a batch of $B$ query-passage pairs $\{(q_i, p_i^+)\}_{i=1}^B$, where $p_i^+$ is the gold passage for query $q_i$, DPR treats all other passages in the batch as negatives, plus additional "hard negatives" retrieved by BM25:

$$\mathcal{L}_{\text{DPR}} = -\log \frac{e^{\text{sim}(q_i, p_i^+)}}{e^{\text{sim}(q_i, p_i^+)} + \sum_{j \neq i} e^{\text{sim}(q_i, p_j^+)} + \sum_{k} e^{\text{sim}(q_i, p_k^-)}}$$

Hard negatives — passages that BM25 ranks highly but that are not actually relevant — are crucial for training effective dense retrievers. Without them, the model learns only to distinguish obviously irrelevant passages.

---

## 15.6 Sparse Retrieval

### 15.6.1 BM25

BM25 (Best Matching 25) is the dominant sparse retrieval algorithm, extending TF-IDF with document length normalization and term frequency saturation (Robertson & Zaragoza, 2009). For a query $Q$ containing terms $q_1, q_2, \ldots, q_n$, the BM25 score for a document $D$ is:

$$\text{BM25}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

where:
- $f(q_i, D)$ is the term frequency of $q_i$ in $D$
- $|D|$ is the document length, $\text{avgdl}$ is the average document length
- $k_1 \in [1.2, 2.0]$ controls term frequency saturation
- $b \in [0, 1]$ controls length normalization (typically $b = 0.75$)
- $\text{IDF}(q_i) = \log \frac{N - n(q_i) + 0.5}{n(q_i) + 0.5}$, where $N$ is the total number of documents and $n(q_i)$ is the number of documents containing $q_i$

BM25 uses **inverted indices** — data structures that map each term to the list of documents containing it, along with term frequencies and positions. This enables sub-linear search time: only documents containing at least one query term are scored.

```python
from rank_bm25 import BM25Okapi
import nltk
nltk.download('punkt')

# Tokenize documents
corpus = [
    "Retrieval-augmented generation improves factual accuracy",
    "Vector databases enable fast similarity search",
    "BM25 is a classical information retrieval algorithm",
    "Dense retrieval uses learned embeddings for search"
]
tokenized_corpus = [doc.lower().split() for doc in corpus]

# Build BM25 index
bm25 = BM25Okapi(tokenized_corpus)

# Query
query = "How does retrieval help with accuracy?"
tokenized_query = query.lower().split()
scores = bm25.get_scores(tokenized_query)
top_docs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
```

### 15.6.2 When Sparse Beats Dense

Despite the success of dense retrieval, BM25 remains competitive — and sometimes superior — in several scenarios:

- **Exact keyword matching:** When queries contain specific entity names, product codes, or technical terms, BM25's lexical matching is more reliable than semantic similarity.
- **Out-of-domain:** Dense retrievers trained on one domain (e.g., Wikipedia) may fail on another (e.g., legal documents). BM25, being unsupervised, generalizes across domains without retraining.
- **Low-resource settings:** BM25 requires no training data, no GPU, and minimal setup.
- **Long-tail queries:** Rare terms have high IDF scores in BM25, making them powerful discriminators. Dense models may not have learned good representations for rare terms.

---

## 15.7 Hybrid Retrieval

### 15.7.1 Combining Dense and Sparse

Hybrid retrieval combines the complementary strengths of dense and sparse methods. Dense retrieval excels at semantic understanding; sparse retrieval excels at exact term matching. Together, they cover failure modes that neither handles alone.

### 15.7.2 Reciprocal Rank Fusion (RRF)

RRF (Cormack et al., 2009) is a simple, effective method for combining ranked lists from multiple retrieval systems. Given ranked lists $R_1, R_2, \ldots, R_m$, the RRF score for document $d$ is:

$$\text{RRF}(d) = \sum_{i=1}^{m} \frac{1}{k + r_i(d)}$$

where $r_i(d)$ is the rank of $d$ in list $R_i$ and $k$ is a constant (typically $k = 60$) that mitigates the impact of high ranks.

```python
def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Combine multiple ranked lists using Reciprocal Rank Fusion."""
    rrf_scores = {}
    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list, start=1):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
            rrf_scores[doc_id] += 1.0 / (k + rank)

    # Sort by RRF score descending
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

# Example usage
dense_results = ["doc_3", "doc_1", "doc_7", "doc_5", "doc_2"]
sparse_results = ["doc_1", "doc_5", "doc_3", "doc_8", "doc_2"]
fused = reciprocal_rank_fusion([dense_results, sparse_results])
```

### 15.7.3 Alpha Weighting

An alternative to RRF is score-level fusion with an alpha parameter:

$$\text{score}(d) = \alpha \cdot \text{score}_{\text{dense}}(d) + (1 - \alpha) \cdot \text{score}_{\text{sparse}}(d)$$

This requires score normalization (e.g., min-max scaling) to bring both scores into the $[0, 1]$ range. The alpha parameter is typically tuned on a validation set; values around $\alpha = 0.7$ (favoring dense) are common starting points.

---

## 15.8 Reranking

### 15.8.1 Cross-Encoders

The bi-encoder architecture, while efficient, processes queries and documents independently — they never "see" each other during encoding. This limits the model's ability to capture fine-grained query-document interactions.

**Cross-encoders** address this by processing the query and document jointly:

$$\text{score}(q, d) = \text{CrossEncoder}([q; \text{SEP}; d])$$

The query and document are concatenated and fed through a single transformer, enabling full cross-attention between query and document tokens. This produces significantly more accurate relevance scores but is computationally expensive — each query-document pair requires a full forward pass.

In practice, cross-encoders are used as **rerankers**: the first stage (bi-encoder or BM25) retrieves a candidate set of 50–100 documents, and the cross-encoder rescores and reorders them.

```python
from sentence_transformers import CrossEncoder

# Load cross-encoder model
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Rerank candidates
query = "What are the benefits of retrieval-augmented generation?"
candidates = [
    "RAG combines retrieval with generation to improve factual accuracy.",
    "Generative adversarial networks are used for image synthesis.",
    "Retrieval systems help ground LLM responses in verified documents.",
    "Fine-tuning adapts models to specific domains."
]

# Score all candidates
pairs = [(query, doc) for doc in candidates]
scores = reranker.predict(pairs)

# Sort by score
ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
for doc, score in ranked:
    print(f"{score:.4f}: {doc}")
```

### 15.8.2 BGE-Reranker and Cohere Rerank

**BGE-Reranker** (from BAAI) is an open-source cross-encoder reranker that achieves strong performance on retrieval benchmarks. The `bge-reranker-v2-m3` model supports multiple languages and can process documents up to 8192 tokens.

**Cohere Rerank** is a commercial API that provides high-quality reranking with a simple API call. It supports multi-language reranking and is easy to integrate:

```python
import cohere

co = cohere.Client("YOUR_API_KEY")

results = co.rerank(
    model="rerank-english-v3.0",
    query="What are the benefits of RAG?",
    documents=candidates,
    top_n=3,
    return_documents=True
)
```

### 15.8.3 ColBERT: Late Interaction

ColBERT (Khattab & Zaharia, 2020) introduces a **late interaction** mechanism that bridges the efficiency of bi-encoders and the effectiveness of cross-encoders.

Instead of compressing each text into a single vector, ColBERT retains **per-token embeddings** for both queries and documents. Relevance is computed via **MaxSim**: for each query token, find the maximum similarity with any document token, then sum across query tokens:

$$\text{score}(q, d) = \sum_{i=1}^{|q|} \max_{j=1}^{|d|} \mathbf{q}_i \cdot \mathbf{d}_j$$

This allows document token embeddings to be precomputed and indexed (like bi-encoders), while still capturing token-level interactions (like cross-encoders). ColBERTv2 adds compression techniques (residual quantization) to reduce storage requirements.

---

## 15.9 RAGatouille: Practical ColBERT for RAG

RAGatouille provides a high-level Python API for using ColBERT in RAG systems:

```python
from ragatouille import RAGPretrainedModel

# Load pretrained ColBERT model
RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

# Index documents
documents = [
    "RAG systems retrieve relevant documents to augment LLM generation.",
    "ColBERT uses late interaction for efficient yet accurate retrieval.",
    "Vector databases store and index high-dimensional embeddings.",
    "BM25 remains competitive for keyword-heavy queries."
]

RAG.index(
    collection=documents,
    index_name="my_index",
    max_document_length=256,
    split_documents=True
)

# Search
results = RAG.search(query="How does late interaction work?", k=3)
for result in results:
    print(f"Score: {result['score']:.4f} | {result['content']}")
```

RAGatouille handles tokenization, index construction, and search in a single interface, making ColBERT-based retrieval accessible to practitioners who do not want to manage the underlying infrastructure.

---

## 15.10 HyDE: Hypothetical Document Embeddings

Gao et al. (2023) proposed **Hypothetical Document Embeddings (HyDE)**, a technique that bridges the lexical gap between short queries and long documents. The intuition is simple: a query and its answer are semantically closer to the relevant documents than the query alone.

The HyDE pipeline:

1. **Generate** a hypothetical answer to the query using an LLM (without any retrieval).
2. **Embed** the hypothetical answer using the document embedding model.
3. **Retrieve** documents similar to the hypothetical answer embedding.

```python
from openai import OpenAI
import numpy as np

client = OpenAI()

def hyde_retrieve(query: str, index, embedding_model, k: int = 5):
    """Retrieve documents using Hypothetical Document Embeddings."""
    # Step 1: Generate hypothetical document
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Write a short passage that answers the question."},
            {"role": "user", "content": query}
        ],
        max_tokens=256
    )
    hypothetical_doc = response.choices[0].message.content

    # Step 2: Embed the hypothetical document
    hyp_embedding = embedding_model.encode([hypothetical_doc])

    # Step 3: Retrieve using hypothetical embedding
    distances, indices = index.search(hyp_embedding, k)
    return indices[0], hypothetical_doc
```

HyDE is particularly effective when queries are short and abstract (e.g., "climate change effects") and the relevant documents are detailed technical passages. It works because the hypothetical answer, even if factually wrong, is likely to use vocabulary and framing similar to the real answer. HyDE can hurt performance when the generated hypothesis is completely off-topic, so it should be used selectively.

---

## 15.11 GraphRAG

### 15.11.1 Motivation and Architecture

Microsoft's **GraphRAG** (Edge et al., 2024) addresses a fundamental limitation of standard RAG: inability to answer questions that require synthesizing information across multiple documents. Standard RAG retrieves chunks independently; GraphRAG builds a knowledge graph that captures relationships between entities across the entire corpus.

The GraphRAG pipeline has two phases:

**Indexing Phase:**
1. **Entity and relationship extraction:** An LLM processes each chunk to extract entities (people, organizations, concepts, locations) and relationships between them.
2. **Knowledge graph construction:** Extracted entities and relationships form a graph $G = (V, E)$.
3. **Community detection:** The Leiden algorithm partitions the graph into hierarchical communities — groups of densely connected entities.
4. **Community summarization:** An LLM generates a summary for each community at each hierarchical level.

**Query Phase:**
- **Local search:** For specific questions, retrieve relevant entities and their neighborhoods from the graph.
- **Global search:** For broad, thematic questions (e.g., "What are the main themes in this dataset?"), aggregate community summaries at an appropriate hierarchical level.

### 15.11.2 When GraphRAG Helps

GraphRAG is most valuable for:
- **Multi-hop questions:** "Which researchers who published on attention mechanisms also worked at Google?"
- **Global summarization:** "What are the major trends in this corpus of 10,000 research papers?"
- **Entity-centric queries:** "Tell me everything about Entity X across all documents."

GraphRAG is more expensive to build (requires many LLM calls during indexing) and is overkill for simple factoid questions. The cost-benefit analysis depends on query patterns: if most queries are multi-hop or thematic, GraphRAG provides significant value.

---

## 15.12 Multimodal RAG

### 15.12.1 CLIP Embeddings for Image Retrieval

CLIP (Radford et al., 2021) provides a shared embedding space for text and images, enabling cross-modal retrieval. In multimodal RAG, both text chunks and images are embedded into this shared space, and queries (text or image) can retrieve either modality.

```python
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Embed text and images into same space
texts = ["A diagram of the RAG architecture", "A photo of a neural network"]
images = [Image.open("rag_diagram.png"), Image.open("neural_net.png")]

# Text embeddings
text_inputs = processor(text=texts, return_tensors="pt", padding=True)
text_embeddings = model.get_text_features(**text_inputs)

# Image embeddings
image_inputs = processor(images=images, return_tensors="pt")
image_embeddings = model.get_image_features(**image_inputs)

# Cross-modal similarity
similarity = torch.nn.functional.cosine_similarity(
    text_embeddings.unsqueeze(1), image_embeddings.unsqueeze(0), dim=-1
)
```

### 15.12.2 Combining Text and Image in Retrieval

A multimodal RAG system indexes both text chunks and images (with captions or OCR-extracted text as metadata). At query time, retrieval searches across both modalities. The retrieved context — a mix of text passages and images — is then passed to a multimodal LLM (GPT-4V, Claude 3.5, Gemini) for generation.

Key design decisions include:
- **Image representation:** Store CLIP embeddings for visual similarity, or generate detailed text descriptions of images and index those?
- **Chunk-image association:** Link images to their surrounding text chunks for context.
- **Generation model:** Must support multimodal input (text + images in the prompt).

---

## 15.13 Long-Context vs. RAG

With models supporting 128K, 200K, or even 1M+ token context windows, a natural question arises: why not just stuff all documents into the context and skip retrieval entirely?

### 15.13.1 When to Use Long Context

Long-context approaches work well when:
- The total corpus fits within the context window.
- You need the model to reason across the *entire* corpus simultaneously.
- Retrieval errors (missing relevant passages) would be costly.
- The task benefits from the model seeing all context (e.g., summarization).

### 15.13.2 When to Use RAG

RAG is preferable when:
- The corpus exceeds any context window (millions of documents).
- Cost per query must be minimized (long contexts are expensive — 1M tokens at $15/M input tokens = $15 per query).
- Latency matters (processing 128K tokens takes seconds; retrieval takes milliseconds).
- You need to cite specific source passages.
- The corpus changes frequently (re-indexing is cheaper than re-prompting).

### 15.13.3 Cost Analysis

| Approach | Corpus: 10 docs | 1K docs | 100K docs | 1M docs |
|----------|-----------------|---------|-----------|---------|
| Long Context | $0.001 | $0.10 | $10.00 | Impossible |
| RAG (top-5) | $0.001 | $0.001 | $0.001 | $0.001 |

The cost advantage of RAG grows linearly with corpus size. However, long-context approaches avoid the failure mode of retrieval — if the retriever misses a relevant passage, the generator cannot use it.

### 15.13.4 Hybrid Approaches

Increasingly, practitioners use a staged approach: RAG retrieves 20–50 relevant chunks, which are then passed as long context to the LLM. This combines the scalability of retrieval with the reasoning capability of long-context models.

---

## 15.14 RAGAS Evaluation

### 15.14.1 The RAGAS Framework

RAGAS (Retrieval Augmented Generation Assessment) provides a comprehensive framework for evaluating RAG systems across four dimensions (Es et al., 2023):

**Context Precision:** Of the retrieved documents, what fraction is actually relevant to the query? High precision means the retriever avoids noise.

$$\text{Context Precision} = \frac{|\text{relevant retrieved}|}{|\text{total retrieved}|}$$

**Context Recall:** Of all relevant documents in the corpus, what fraction did the retriever find? High recall means the retriever does not miss important information.

$$\text{Context Recall} = \frac{|\text{relevant retrieved}|}{|\text{total relevant}|}$$

**Faithfulness:** Is the generated answer supported by the retrieved context? This measures hallucination — does the model make claims not grounded in the provided documents?

**Answer Relevance:** Does the generated answer actually address the query? A faithful answer that does not address the question is still a failure.

### 15.14.2 Implementation

```python
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy
)
from datasets import Dataset

# Prepare evaluation dataset
eval_data = {
    "question": [
        "What is retrieval-augmented generation?",
        "How does BM25 work?"
    ],
    "answer": [
        "RAG combines document retrieval with LLM generation to improve accuracy.",
        "BM25 scores documents using term frequency with length normalization."
    ],
    "contexts": [
        ["RAG retrieves relevant documents and uses them as context for generation."],
        ["BM25 extends TF-IDF with document length normalization and saturation."]
    ],
    "ground_truth": [
        "RAG retrieves documents and augments LLM prompts with them for grounded generation.",
        "BM25 is a ranking function using term frequency, inverse document frequency, and length normalization."
    ]
}

dataset = Dataset.from_dict(eval_data)

# Evaluate
results = evaluate(
    dataset=dataset,
    metrics=[context_precision, context_recall, faithfulness, answer_relevancy]
)

print(results)
```

### 15.14.3 Beyond RAGAS

Additional evaluation strategies include:
- **End-to-end accuracy:** On QA benchmarks, does the RAG system answer correctly more often than the LLM alone?
- **Retrieval-specific metrics:** MRR (Mean Reciprocal Rank), NDCG (Normalized Discounted Cumulative Gain), Recall@K.
- **Human evaluation:** Domain experts judge answer quality, groundedness, and completeness.
- **LLM-as-judge:** Use a powerful LLM (e.g., GPT-4) to evaluate generated answers against reference answers, providing scalable evaluation.

---

## 15.15 Production RAG Architecture

### 15.15.1 Caching

Caching is essential for reducing latency and cost in production RAG systems.

- **Semantic cache:** Hash the query embedding and cache results for similar queries. If a new query's embedding is within a cosine distance threshold of a cached query, return the cached result.
- **Exact cache:** Cache exact query-answer pairs for frequently asked questions.
- **Embedding cache:** Cache document embeddings to avoid recomputation during index rebuilds.

### 15.15.2 Query Routing

Not all queries require retrieval. A query router classifies incoming queries and routes them appropriately:

- **Direct LLM response:** For general knowledge questions, greetings, or math problems.
- **RAG pipeline:** For domain-specific questions requiring grounding.
- **SQL/structured query:** For questions about structured data.
- **Clarification:** For ambiguous queries.

```python
def route_query(query: str, classifier) -> str:
    """Route a query to the appropriate handler."""
    category = classifier.predict(query)

    if category == "general_knowledge":
        return "llm_direct"
    elif category == "domain_specific":
        return "rag_pipeline"
    elif category == "data_query":
        return "sql_pipeline"
    else:
        return "clarification"
```

### 15.15.3 Monitoring and Observability

Production RAG systems require monitoring at every stage:

- **Retrieval quality:** Track retrieval scores, number of retrieved documents, and relevance signals.
- **Generation quality:** Monitor faithfulness scores, answer length, and user feedback (thumbs up/down).
- **Latency breakdown:** Measure time spent in embedding, retrieval, reranking, and generation.
- **Cost tracking:** Monitor token usage, API costs, and infrastructure costs.

Tools like **LangSmith**, **Arize Phoenix**, and **Weights & Biases** provide observability platforms specifically designed for LLM and RAG systems.

### 15.15.4 Embedding Versioning

When embedding models are updated, all documents must be re-embedded — old and new embeddings are incompatible. Production systems should:
- Version embedding indices alongside the model that produced them.
- Support blue-green deployments to switch between embedding versions without downtime.
- Track which embedding model was used for each indexed document.

---

## 15.16 LangChain and LlamaIndex Integration

### 15.16.1 LangChain RAG Pipeline

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 1. Load documents
loader = PyPDFLoader("research_paper.pdf")
documents = loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_documents(documents)

# 3. Create embeddings and vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(chunks, embeddings)

# 4. Create retrieval chain
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

prompt_template = PromptTemplate(
    template="""Use the following context to answer the question.
If you don't know the answer based on the context, say so.

Context: {context}

Question: {question}

Answer:""",
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o", temperature=0),
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt_template},
    return_source_documents=True
)

# 5. Query
result = qa_chain.invoke({"query": "What are the main findings of this paper?"})
print(result["result"])
for doc in result["source_documents"]:
    print(f"Source: {doc.metadata['source']}, Page: {doc.metadata['page']}")
```

### 15.16.2 LlamaIndex RAG Pipeline

```python
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SentenceSplitter

# Configure global settings
Settings.llm = OpenAI(model="gpt-4o", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# 1. Load documents
documents = SimpleDirectoryReader("./data/").load_data()

# 2. Parse into nodes (chunks)
parser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
nodes = parser.get_nodes_from_documents(documents)

# 3. Build index
index = VectorStoreIndex(nodes)

# 4. Create query engine
query_engine = index.as_query_engine(
    similarity_top_k=5,
    response_mode="tree_summarize"  # Hierarchical summarization of retrieved chunks
)

# 5. Query
response = query_engine.query("What are the main findings of this paper?")
print(response)

# Access source nodes
for node in response.source_nodes:
    print(f"Score: {node.score:.4f}")
    print(f"Text: {node.text[:200]}...")
```

LlamaIndex's key differentiator is its rich set of **response modes** — `refine` (iteratively refine an answer with each chunk), `tree_summarize` (build a summary tree from chunks), and `compact` (stuff as many chunks as possible before refining) — and its native support for structured data through SQL and knowledge graph indices.

---

## Summary

RAG has emerged as the dominant paradigm for grounding LLM outputs in external knowledge. This chapter has traced the full pipeline from document ingestion through chunking, embedding, indexing, retrieval (dense, sparse, and hybrid), reranking, and generation. We explored advanced techniques — HyDE for bridging the query-document gap, GraphRAG for multi-hop reasoning, and multimodal RAG for cross-modal retrieval — and examined evaluation with the RAGAS framework. The chapter concluded with production considerations: caching, routing, monitoring, and framework integration with LangChain and LlamaIndex.

The key insight is that RAG is not a single technique but a *system design problem*. The choice of chunking strategy, embedding model, index type, retrieval method, and reranking approach depends on the specific requirements of the application — its corpus size, query patterns, latency budget, and accuracy requirements. There is no one-size-fits-all RAG architecture; the best systems are carefully tuned to their domain.

---

## Exercises

1. **Chunking comparison.** Take a 50-page PDF document and implement three chunking strategies: fixed-size (500 tokens, 100 overlap), recursive character splitting, and semantic chunking. Build a RAG pipeline with each and compare retrieval quality on 20 manually crafted questions.

2. **Embedding benchmark.** Compare three embedding models — `all-MiniLM-L6-v2`, `bge-large-en-v1.5`, and `text-embedding-3-small` — on the same corpus. Measure Recall@5, Recall@10, and average query latency.

3. **Hybrid retrieval.** Implement a hybrid retrieval system combining BM25 and dense retrieval with RRF. Compare its performance to dense-only and sparse-only baselines on a question-answering dataset.

4. **Reranking ablation.** Add a cross-encoder reranker to your RAG pipeline. Measure the change in RAGAS faithfulness and answer relevance scores with and without reranking.

5. **HyDE implementation.** Implement HyDE and test it on queries of varying length and specificity. Identify cases where HyDE helps and cases where it hurts.

6. **Cost analysis.** For a corpus of 100,000 documents averaging 2,000 tokens each, calculate the cost of (a) embedding the entire corpus with `text-embedding-3-small`, (b) storing the index in Pinecone, (c) processing 10,000 queries per day with RAG (top-5 retrieval + GPT-4o generation), and (d) processing the same queries using long-context stuffing.

7. **Production architecture.** Design a production RAG system for a customer support chatbot that serves 1,000 queries per minute. Specify your choices for embedding model, vector database, retrieval strategy, caching layer, and monitoring. Justify each decision.

---

## References

Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009). Reciprocal rank fusion outperforms condorcet and individual rank learning methods. *Proceedings of the 32nd International ACM SIGIR Conference on Research and Development in Information Retrieval*, 758–759.

Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., ... & Larson, J. (2024). From local to global: A graph RAG approach to query-focused summarization. *arXiv preprint arXiv:2404.16130*.

Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2023). RAGAS: Automated evaluation of retrieval augmented generation. *arXiv preprint arXiv:2309.15217*.

Gao, L., Ma, X., Lin, J., & Callan, J. (2023). Precise zero-shot dense retrieval without relevance labels. *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics*, 1762–1777.

Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. *Advances in Neural Information Processing Systems*, 33, 6840–6851.

Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., ... & Fung, P. (2023). Survey of hallucination in natural language generation. *ACM Computing Surveys*, 55(12), 1–38.

Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity search with GPUs. *IEEE Transactions on Big Data*, 7(3), 535–547.

Karpukhin, V., Oguz, B., Min, S., Lewis, P., Wu, L., Edunov, S., ... & Yih, W. (2020). Dense passage retrieval for open-domain question answering. *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing*, 6769–6781.

Khattab, O., & Zaharia, M. (2020). ColBERT: Efficient and effective passage search via contextualized late interaction over BERT. *Proceedings of the 43rd International ACM SIGIR Conference*, 39–48.

Kusupati, A., Bhatt, G., Rege, A., Wallingford, M., Sinha, A., Ramanujan, V., ... & Farhadi, A. (2022). Matryoshka representation learning. *Advances in Neural Information Processing Systems*, 35, 30233–30249.

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., ... & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems*, 33, 9459–9474.

Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). Distributed representations of words and phrases and their compositionality. *Advances in Neural Information Processing Systems*, 26.

Oord, A. v. d., Li, Y., & Vinyals, O. (2018). Representation learning with contrastive predictive coding. *arXiv preprint arXiv:1807.03748*.

Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global vectors for word representation. *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing*, 1532–1543.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., ... & Sutskever, I. (2021). Learning transferable visual models from natural language supervision. *Proceedings of the 38th International Conference on Machine Learning*, 8748–8763.

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing*, 3982–3992.

Robertson, S., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in Information Retrieval*, 3(4), 333–389.

Wang, L., Yang, N., Huang, X., Jiao, B., Yang, L., Jiang, D., ... & Wei, F. (2022). Text embeddings by weakly-supervised contrastive pre-training. *arXiv preprint arXiv:2212.03533*.
