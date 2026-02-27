# Chapter 24: Graph Machine Learning

> *"Graphs are the most general data structure. Images are grids, text is a sequence, molecules are graphs, social networks are graphs, the Web is a graph. Once you learn to think in graphs, you see them everywhere."*

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Represent graph-structured data using adjacency matrices, edge lists, and node/edge feature matrices, and distinguish homogeneous from heterogeneous graphs.
2. Explain the message passing neural network (MPNN) framework and derive how GNNs aggregate neighborhood information.
3. Implement Graph Convolutional Networks (GCN), GraphSAGE, and Graph Attention Networks (GAT) and articulate their tradeoffs.
4. Build graph learning pipelines with PyTorch Geometric, including data loading, batching, and training loops.
5. Understand Graph Transformers (Graphormer) and how they combine attention mechanisms with graph structure.
6. Represent knowledge graphs and apply embedding methods (TransE, RotatE, DistMult, ComplEx) for link prediction.
7. Model temporal/dynamic graphs using Temporal Graph Networks (TGN) for evolving relationships.
8. Apply graph neural networks to molecular property prediction, molecular generation, and protein-ligand docking.
9. Generate new graphs using autoregressive, VAE-based, and diffusion-based methods.

---

## 24.1 Graph Fundamentals

Graphs are mathematical structures that model pairwise relationships between entities. A graph $G = (V, E)$ consists of a set of nodes (vertices) $V$ and edges $E \subseteq V \times V$ connecting them. What makes graphs particularly challenging for machine learning is their irregular structure — unlike images (regular grids) or text (sequences), graphs have no fixed topology.

### 24.1.1 Graph Representations

**Adjacency Matrix.** An $n \times n$ matrix $\mathbf{A}$ where $A_{ij} = 1$ if edge $(i, j) \in E$, and $A_{ij} = 0$ otherwise. For undirected graphs, $\mathbf{A}$ is symmetric. For weighted graphs, $A_{ij}$ holds the edge weight.

$$\mathbf{A} = \begin{pmatrix} 0 & 1 & 0 & 1 \\ 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 1 & 0 & 1 & 0 \end{pmatrix}$$

Memory: $O(|V|^2)$. Efficient for dense graphs but wasteful for sparse ones.

**Edge List.** A list of $(i, j)$ pairs: $\{(0, 1), (0, 3), (1, 2), (2, 3)\}$. Memory: $O(|E|)$. Preferred for sparse graphs.

**Adjacency List.** For each node, store its neighbors: $\{0: [1, 3], 1: [0, 2], 2: [1, 3], 3: [0, 2]\}$. Memory: $O(|V| + |E|)$. Efficient for neighborhood queries.

### 24.1.2 Node, Edge, and Graph Features

Graphs in machine learning are typically *attributed* — nodes, edges, and the graph itself carry feature vectors:

- **Node features** $\mathbf{X} \in \mathbb{R}^{|V| \times d}$: Feature vector for each node (e.g., user profile, atom type).
- **Edge features** $\mathbf{E} \in \mathbb{R}^{|E| \times d_e}$: Feature vector for each edge (e.g., relationship type, bond type).
- **Graph-level features** $\mathbf{u} \in \mathbb{R}^{d_g}$: Global features of the entire graph (e.g., molecular weight).

### 24.1.3 Homogeneous vs Heterogeneous Graphs

**Homogeneous graphs** have a single type of node and edge. Examples: citation networks (paper cites paper), social networks (person knows person).

**Heterogeneous graphs** have multiple node and/or edge types. Examples:
- Academic graphs: authors *write* papers, papers *cite* papers, papers *belong to* venues.
- E-commerce: users *purchase* products, products *belong to* categories, users *review* products.

Heterogeneous graphs require type-aware message passing, where different node/edge types have different transformation functions.

### 24.1.4 Common Graph Tasks

**Node classification.** Predict labels for individual nodes (e.g., categorize users in a social network, classify proteins by function).

**Link prediction.** Predict whether an edge exists between two nodes (e.g., recommend friends, predict drug-target interactions, complete knowledge graphs).

**Graph classification.** Predict a label for the entire graph (e.g., predict molecular toxicity, classify documents represented as graphs).

**Graph regression.** Predict a continuous value for the entire graph (e.g., predict molecular binding affinity).

**Node/graph clustering.** Discover communities or functional groups in the graph.

---

## 24.2 Message Passing Neural Networks

The Message Passing Neural Network (MPNN) framework (Gilmer et al., 2017) provides a unified formulation for most graph neural networks. The core idea: each node iteratively updates its representation by *passing messages* with its neighbors.

### 24.2.1 General Framework

At each layer $\ell$, the MPNN computes:

1. **Message computation:** For each edge $(j, i)$, compute a message from node $j$ to node $i$:

$$\mathbf{m}_{j \to i}^{(\ell)} = M^{(\ell)}(\mathbf{h}_i^{(\ell-1)}, \mathbf{h}_j^{(\ell-1)}, \mathbf{e}_{ji})$$

where $\mathbf{h}_i^{(\ell-1)}$ is node $i$'s representation at layer $\ell - 1$, and $\mathbf{e}_{ji}$ is the edge feature.

2. **Aggregation:** Aggregate messages from all neighbors of node $i$ using a permutation-invariant function:

$$\mathbf{m}_i^{(\ell)} = \bigoplus_{j \in \mathcal{N}(i)} \mathbf{m}_{j \to i}^{(\ell)}$$

where $\bigoplus$ is a reduction operator such as $\sum$, $\text{mean}$, or $\max$.

3. **Update:** Update the node representation:

$$\mathbf{h}_i^{(\ell)} = U^{(\ell)}(\mathbf{h}_i^{(\ell-1)}, \mathbf{m}_i^{(\ell)})$$

After $L$ layers, node $i$'s representation $\mathbf{h}_i^{(L)}$ encodes information from its $L$-hop neighborhood. For graph-level tasks, a *readout* function aggregates all node representations:

$$\mathbf{h}_G = R(\{\mathbf{h}_i^{(L)} | i \in V\})$$

Common readout functions include mean pooling, sum pooling, attention-weighted sum, and virtual node approaches.

### 24.2.2 Expressiveness and the WL Test

The Weisfeiler-Leman (WL) graph isomorphism test provides a theoretical framework for analyzing GNN expressiveness. Xu et al. (2019) proved that standard MPNNs are at most as powerful as the 1-WL test in distinguishing non-isomorphic graphs. This means there exist structurally different graphs that no standard MPNN can distinguish. This motivates more expressive architectures like higher-order GNNs and graph transformers.

---

## 24.3 Graph Convolutional Networks (GCN)

The Graph Convolutional Network (Kipf & Welling, 2017) is the foundational GNN architecture, derived from spectral graph theory.

### 24.3.1 Spectral Approach

Graph convolution is defined in the spectral domain using the graph Laplacian $\mathbf{L} = \mathbf{D} - \mathbf{A}$, where $\mathbf{D}$ is the degree matrix. The normalized Laplacian is $\mathbf{L}_\text{norm} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2}$.

The eigendecomposition $\mathbf{L}_\text{norm} = \mathbf{U} \boldsymbol{\Lambda} \mathbf{U}^T$ defines the graph Fourier transform. A spectral convolution is:

$$\mathbf{x} *_G \mathbf{g} = \mathbf{U} \, g(\boldsymbol{\Lambda}) \, \mathbf{U}^T \mathbf{x}$$

where $g(\boldsymbol{\Lambda})$ is a filter in the spectral domain. Computing this directly is $O(n^3)$ due to the eigendecomposition.

### 24.3.2 Kipf & Welling Simplification

ChebNet (Defferrard et al., 2016) approximated spectral filters with Chebyshev polynomials. Kipf and Welling simplified further by using a first-order approximation ($K = 1$ Chebyshev polynomial) and adding a re-normalization trick:

$$\mathbf{H}^{(\ell+1)} = \sigma\left( \tilde{\mathbf{D}}^{-1/2} \tilde{\mathbf{A}} \tilde{\mathbf{D}}^{-1/2} \mathbf{H}^{(\ell)} \mathbf{W}^{(\ell)} \right)$$

where:
- $\tilde{\mathbf{A}} = \mathbf{A} + \mathbf{I}$ adds self-loops (each node aggregates its own features).
- $\tilde{\mathbf{D}}_{ii} = \sum_j \tilde{A}_{ij}$ is the degree matrix of $\tilde{\mathbf{A}}$.
- $\mathbf{W}^{(\ell)} \in \mathbb{R}^{d_\ell \times d_{\ell+1}}$ is a learnable weight matrix.
- $\sigma$ is a nonlinearity (typically ReLU).

The normalization $\tilde{\mathbf{D}}^{-1/2} \tilde{\mathbf{A}} \tilde{\mathbf{D}}^{-1/2}$ is the symmetric normalization, which prevents the scale of feature vectors from changing with node degree.

**Per-node formulation:**

$$\mathbf{h}_i^{(\ell+1)} = \sigma\left( \sum_{j \in \mathcal{N}(i) \cup \{i\}} \frac{1}{\sqrt{\tilde{d}_i \tilde{d}_j}} \mathbf{h}_j^{(\ell)} \mathbf{W}^{(\ell)} \right)$$

where $\tilde{d}_i = |\mathcal{N}(i)| + 1$ (degree including self-loop).

### 24.3.3 The Over-Smoothing Problem

A critical limitation of GCNs is *over-smoothing*: as the number of layers increases, node representations converge to the same value, losing discriminative power. After $L$ layers, each node's representation depends on its $L$-hop neighborhood. For small-world graphs, a few layers already reach the entire graph, making all representations nearly identical.

Mitigations include:
- **Residual connections:** $\mathbf{h}^{(\ell+1)} = \mathbf{h}^{(\ell)} + \text{GNN}(\mathbf{h}^{(\ell)})$
- **Jumping Knowledge (JK) Networks:** Concatenate or select representations from all layers.
- **PairNorm / NodeNorm:** Normalization techniques that prevent convergence.
- **DropEdge:** Randomly remove edges during training.

---

## 24.4 GraphSAGE

GraphSAGE (Graph SAmple and aggreGatE) by Hamilton et al. (2017) addresses two limitations of GCN: it enables *inductive* learning (generalizing to unseen nodes/graphs) and scales to large graphs through neighborhood *sampling*.

### 24.4.1 Inductive Learning

GCN is *transductive* — it operates on the full graph Laplacian and cannot generalize to new nodes without retraining. GraphSAGE learns aggregation *functions* rather than fixed embeddings, enabling prediction on nodes not seen during training.

### 24.4.2 Sampling and Aggregating

For each node, GraphSAGE:
1. **Samples** a fixed number of neighbors (e.g., $K_1 = 25$ from 1-hop, $K_2 = 10$ from 2-hop) rather than using all neighbors.
2. **Aggregates** neighbor features using a learned aggregation function.
3. **Combines** the aggregated neighborhood with the node's own features.

$$\mathbf{h}_{\mathcal{N}(i)}^{(\ell)} = \text{AGG}^{(\ell)}\left(\{\mathbf{h}_j^{(\ell-1)} : j \in \mathcal{N}_s(i)\}\right)$$
$$\mathbf{h}_i^{(\ell)} = \sigma\left( \mathbf{W}^{(\ell)} \cdot [\mathbf{h}_i^{(\ell-1)} \| \mathbf{h}_{\mathcal{N}(i)}^{(\ell)}] \right)$$

where $\mathcal{N}_s(i)$ is a sampled subset of neighbors and $\|$ denotes concatenation.

### 24.4.3 Aggregator Variants

- **Mean aggregator:** $\text{AGG} = \text{mean}(\{\mathbf{h}_j\})$. Equivalent to GCN (without degree normalization).
- **LSTM aggregator:** Apply an LSTM to a random permutation of neighbor features. Theoretically not permutation-invariant, but works well in practice.
- **Pool aggregator:** $\text{AGG} = \max(\{\sigma(\mathbf{W}_\text{pool} \mathbf{h}_j + \mathbf{b})\})$. Apply a pointwise MLP then element-wise max pooling.

The pooling aggregator is generally preferred for its expressiveness.

---

## 24.5 Graph Attention Networks (GAT)

Graph Attention Networks (Velickovic et al., 2018) introduce learnable *attention coefficients* that weight neighbor contributions, allowing the model to focus on the most important neighbors for each node.

### 24.5.1 Attention Mechanism

For each edge $(j, i)$, GAT computes an attention coefficient:

$$e_{ij} = \text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W} \mathbf{h}_i \| \mathbf{W} \mathbf{h}_j]\right)$$

where $\mathbf{W} \in \mathbb{R}^{d' \times d}$ is a shared linear transformation and $\mathbf{a} \in \mathbb{R}^{2d'}$ is the attention vector.

The coefficients are normalized across the neighborhood using softmax:

$$\alpha_{ij} = \text{softmax}_j(e_{ij}) = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}$$

The node update is the attention-weighted sum:

$$\mathbf{h}_i^{(\ell+1)} = \sigma\left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij} \mathbf{W} \mathbf{h}_j^{(\ell)} \right)$$

### 24.5.2 Multi-Head Attention

GAT uses multi-head attention for stability and expressiveness. $K$ independent attention heads compute different attention patterns, and their outputs are concatenated (in intermediate layers) or averaged (in the final layer):

$$\mathbf{h}_i^{(\ell+1)} = \Big\|_{k=1}^{K} \sigma\left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^k \mathbf{W}^k \mathbf{h}_j^{(\ell)} \right)$$

This is analogous to multi-head attention in Transformers. With 8 heads and $d' = 8$ features per head, GAT produces 64-dimensional output representations.

### 24.5.3 GATv2

Brody et al. (2022) identified a limitation of GAT's attention mechanism: it computes *static* attention where the ranking of attention scores is the same regardless of the query node. GATv2 fixes this by applying the nonlinearity after concatenation:

$$e_{ij} = \mathbf{a}^T \text{LeakyReLU}\left(\mathbf{W} [\mathbf{h}_i \| \mathbf{h}_j]\right)$$

This produces *dynamic* attention where the attention ranking depends on both the query and key nodes.

---

## 24.6 PyTorch Geometric

PyTorch Geometric (PyG) by Fey and Lenssen (2019) is the leading library for deep learning on graphs and other irregular structures.

### 24.6.1 Data Object

The fundamental data structure is `torch_geometric.data.Data`:

```python
import torch
from torch_geometric.data import Data

# Create a graph with 4 nodes and 4 edges
edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 3, 3, 0],  # source nodes
    [1, 0, 2, 1, 3, 2, 0, 3]   # target nodes
], dtype=torch.long)

# Node features (4 nodes, 3 features each)
x = torch.tensor([
    [1.0, 0.0, 0.5],
    [0.0, 1.0, 0.3],
    [1.0, 1.0, 0.8],
    [0.0, 0.0, 0.1]
], dtype=torch.float)

# Node labels (for node classification)
y = torch.tensor([0, 1, 1, 0], dtype=torch.long)

data = Data(x=x, edge_index=edge_index, y=y)

print(data)
# Data(x=[4, 3], edge_index=[2, 8], y=[4])
print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}")
print(f"Isolated: {data.has_isolated_nodes()}")
print(f"Self-loops: {data.has_self_loops()}")
print(f"Undirected: {data.is_undirected()}")
```

### 24.6.2 DataLoader and Batching

PyG batches multiple graphs into a single disconnected graph using block-diagonal adjacency:

```python
from torch_geometric.loader import DataLoader

# Assume dataset is a list of Data objects (e.g., molecules)
dataset = [...]  # list of Data objects
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in loader:
    # batch.x: all node features stacked (sum of nodes across batch)
    # batch.edge_index: all edges, with indices offset per graph
    # batch.batch: mapping from each node to its graph index
    # batch.y: graph-level labels
    print(f"Batch: {batch.num_graphs} graphs, {batch.num_nodes} total nodes")
```

### 24.6.3 MessagePassing Base Class

PyG provides the `MessagePassing` class for implementing custom GNN layers:

```python
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class GCNConv(MessagePassing):
    """Graph Convolutional Network layer (Kipf & Welling, 2017)."""
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # Sum aggregation
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x, edge_index):
        # Add self-loops: A_hat = A + I
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Compute normalization: D^(-1/2)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Linear transformation
        x = self.linear(x)

        # Message passing
        out = self.propagate(edge_index, x=x, norm=norm)
        return out + self.bias

    def message(self, x_j, norm):
        # x_j: features of source nodes (neighbors)
        # norm: normalization coefficients
        return norm.view(-1, 1) * x_j
```

### 24.6.4 Complete Node Classification Example

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

# Load Cora citation network
dataset = Planetoid(root='/tmp/Cora', name='Cora',
                    transform=NormalizeFeatures())
data = dataset[0]

print(f"Dataset: {dataset}")
print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}")
print(f"Features: {data.num_node_features}")
print(f"Classes: {dataset.num_classes}")
print(f"Train/Val/Test: {data.train_mask.sum()}/{data.val_mask.sum()}"
      f"/{data.test_mask.sum()}")


class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=64, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GAT(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=8,
                 heads=8, dropout=0.6):
        super().__init__()
        self.conv1 = GATConv(num_features, hidden_dim, heads=heads,
                             dropout=dropout)
        self.conv2 = GATConv(hidden_dim * heads, num_classes, heads=1,
                             concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


def train_and_evaluate(model, data, lr=0.01, weight_decay=5e-4, epochs=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)

    best_val_acc = 0
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            out = model(data)
            pred = out.argmax(dim=1)

            train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean()
            val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean()
            test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss={loss:.4f}, Train={train_acc:.4f}, "
                  f"Val={val_acc:.4f}, Test={test_acc:.4f}")

    print(f"\nBest Test Accuracy: {best_test_acc:.4f}")
    return best_test_acc


# Train GCN
print("=== GCN ===")
gcn_model = GCN(dataset.num_features, dataset.num_classes)
train_and_evaluate(gcn_model, data)

# Train GAT
print("\n=== GAT ===")
gat_model = GAT(dataset.num_features, dataset.num_classes)
train_and_evaluate(gat_model, data)
```

---

## 24.7 Graph Transformers

Standard GNNs are limited by the message passing paradigm — information propagates one hop per layer, and expressiveness is bounded by the WL test. Graph Transformers address these limitations by applying transformer-style attention to graphs.

### 24.7.1 Graphormer

Graphormer (Ying et al., 2021) adapts the Transformer architecture for graph-structured data by encoding structural information through three bias terms added to the attention scores:

**Centrality Encoding.** Nodes with higher centrality (degree) are given distinct embeddings:

$$\mathbf{h}_i^{(0)} = \mathbf{x}_i + \mathbf{z}^-_{\text{deg}^-(i)} + \mathbf{z}^+_{\text{deg}^+(i)}$$

where $\mathbf{z}^-$ and $\mathbf{z}^+$ are learnable embeddings indexed by in-degree and out-degree.

**Spatial Encoding.** The shortest-path distance $d(i, j)$ between nodes $i$ and $j$ is used as a bias in attention:

$$A_{ij} = \frac{(\mathbf{h}_i \mathbf{W}_Q)(\mathbf{h}_j \mathbf{W}_K)^T}{\sqrt{d}} + b_{d(i,j)}$$

where $b_{d(i,j)}$ is a learnable scalar bias for each distance.

**Edge Encoding.** Features of edges along the shortest path between two nodes are incorporated:

$$c_{ij} = \frac{1}{L} \sum_{\ell=1}^{L} \mathbf{e}_{n_\ell, n_{\ell+1}}^T \mathbf{w}_\ell$$

where $\mathbf{e}_{n_\ell, n_{\ell+1}}$ are edge features along the shortest path and $\mathbf{w}_\ell$ are learnable weights.

**Advantages over MPNNs.** Graphormer computes attention over all pairs of nodes (not just neighbors), enabling long-range interactions in a single layer. Combined with structural encodings, it achieves strong performance on molecular property prediction benchmarks (e.g., OGB-LSC).

### 24.7.2 Graph-BERT

Zhang et al. (2020) proposed Graph-BERT, which samples local subgraphs and applies BERT-style pretraining with graph-specific tasks:
- **Node attribute reconstruction:** Mask node features and predict them.
- **Graph structure recovery:** Predict the adjacency matrix from node embeddings.

### 24.7.3 Combining Attention with Graph Structure

A central design question for graph transformers is how to balance global attention (which enables long-range dependencies but loses structural information) with local message passing (which respects graph topology but is limited in range). Recent approaches include:

- **GPS (General, Powerful, Scalable):** Interleaves local message passing (MPNN) layers with global attention layers, combining the strengths of both.
- **Exphormer:** Uses virtual nodes and expander graph sparsification to achieve efficient global attention.
- **NodeFormer:** Uses random feature maps (kernelized attention) for linear-complexity all-to-all message passing.

---

## 24.8 Knowledge Graphs

Knowledge graphs represent factual knowledge as collections of (subject, predicate, object) triples — e.g., (Paris, capital_of, France). They are used for information retrieval, question answering, and recommendation systems.

### 24.8.1 Formalization

A knowledge graph $\mathcal{G} = (\mathcal{E}, \mathcal{R}, \mathcal{T})$ consists of:
- **Entities** $\mathcal{E}$: Objects in the world (e.g., Paris, France).
- **Relations** $\mathcal{R}$: Types of relationships (e.g., capital_of, located_in).
- **Triples** $\mathcal{T} \subseteq \mathcal{E} \times \mathcal{R} \times \mathcal{E}$: Facts, e.g., $(h, r, t)$ where $h$ is the head entity, $r$ is the relation, and $t$ is the tail entity.

Major knowledge graphs include Freebase, Wikidata (90+ million items), DBpedia, and domain-specific ones like DrugBank (drug-gene interactions).

### 24.8.2 Knowledge Graph Embedding Methods

KG embedding methods learn low-dimensional vectors for entities and relations, defining a scoring function $f(h, r, t)$ that is high for true triples and low for false ones.

**TransE (Bordes et al., 2013).** Models relations as translations in the embedding space:

$$f(h, r, t) = -\|\mathbf{h} + \mathbf{r} - \mathbf{t}\|$$

The intuition: if $(h, r, t)$ is a true triple, then $\mathbf{h} + \mathbf{r} \approx \mathbf{t}$. Simple and effective but cannot model symmetric relations ($r(a, b) \wedge r(b, a)$) or 1-to-N relations well (because $\mathbf{h} + \mathbf{r}$ maps to a single point, not multiple).

**RotatE (Sun et al., 2019).** Models relations as rotations in complex space:

$$f(h, r, t) = -\|\mathbf{h} \circ \mathbf{r} - \mathbf{t}\|$$

where $\mathbf{h}, \mathbf{r}, \mathbf{t} \in \mathbb{C}^d$, $|\mathbf{r}_i| = 1$ (unit complex numbers representing rotations), and $\circ$ is the Hadamard (element-wise) product. RotatE can model symmetry ($\mathbf{r} \circ \mathbf{r} = \mathbf{1}$), antisymmetry, inversion, and composition patterns.

**DistMult (Yang et al., 2015).** Uses a bilinear scoring function:

$$f(h, r, t) = \mathbf{h}^T \text{diag}(\mathbf{r}) \mathbf{t} = \sum_i h_i \cdot r_i \cdot t_i$$

Simple and efficient but symmetric ($f(h, r, t) = f(t, r, h)$), so it cannot model asymmetric relations like "parent_of".

**ComplEx (Trouillon et al., 2016).** Extends DistMult to complex space:

$$f(h, r, t) = \text{Re}\left(\sum_i h_i \cdot r_i \cdot \bar{t}_i\right)$$

where $\bar{t}_i$ is the complex conjugate. The use of complex numbers breaks the symmetry, allowing ComplEx to model both symmetric and antisymmetric relations.

### 24.8.3 Link Prediction

The primary task on knowledge graphs is link prediction: given a query $(h, r, ?)$ or $(?, r, t)$, rank all candidate entities by their score. Evaluation metrics include:

- **Mean Reciprocal Rank (MRR):** Average of $\frac{1}{\text{rank}}$ across queries.
- **Hits@k:** Proportion of queries where the true entity is in the top $k$.
- **Filtered setting:** Excludes other known true triples from the ranking (standard practice).

```python
import torch
import torch.nn as nn

class TransE(nn.Module):
    """TransE knowledge graph embedding model."""
    def __init__(self, num_entities, num_relations, embedding_dim=100, margin=1.0):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, embedding_dim)
        self.relation_emb = nn.Embedding(num_relations, embedding_dim)
        self.margin = margin

        # Initialize embeddings
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        # Normalize relation embeddings
        with torch.no_grad():
            self.relation_emb.weight.data = F.normalize(
                self.relation_emb.weight.data, dim=1
            )

    def score(self, head, relation, tail):
        """Compute score for (head, relation, tail) triples."""
        h = self.entity_emb(head)
        r = self.relation_emb(relation)
        t = self.entity_emb(tail)
        # Normalize entity embeddings
        h = F.normalize(h, dim=-1)
        t = F.normalize(t, dim=-1)
        return -torch.norm(h + r - t, p=2, dim=-1)

    def loss(self, pos_head, pos_rel, pos_tail, neg_head, neg_rel, neg_tail):
        """Margin-based ranking loss."""
        pos_score = self.score(pos_head, pos_rel, pos_tail)
        neg_score = self.score(neg_head, neg_rel, neg_tail)
        return torch.relu(self.margin - pos_score + neg_score).mean()
```

---

## 24.9 Temporal Graphs

Real-world graphs evolve over time — new nodes and edges appear, existing ones change or disappear. Temporal graph learning models these dynamics.

### 24.9.1 Types of Dynamic Graphs

- **Discrete-time dynamic graphs (DTDG):** A sequence of graph snapshots $\{G^{(1)}, G^{(2)}, \ldots, G^{(T)}\}$ at fixed time intervals.
- **Continuous-time dynamic graphs (CTDG):** Events (edge additions/deletions, node changes) occur at continuous timestamps.

### 24.9.2 Temporal Random Walks

Extending random walks (as in node2vec) to temporal graphs requires respecting temporal order: walks can only traverse edges with non-decreasing timestamps. This captures temporal causality — a recommendation depends on past interactions, not future ones.

### 24.9.3 Temporal Graph Networks (TGN)

TGN (Rossi et al., 2020) is a framework for learning on continuous-time dynamic graphs. For each event (e.g., an interaction between nodes $u$ and $v$ at time $t$), TGN:

1. **Message function:** Computes a message from the event:
$$\mathbf{m}_u(t) = \text{msg}(\mathbf{s}_u(t^-), \mathbf{s}_v(t^-), \Delta t, \mathbf{e}_{uv}(t))$$

2. **Message aggregation:** If multiple events involve the same node, aggregate their messages.

3. **Memory update:** Update the node's memory (a latent state vector maintained for each node):
$$\mathbf{s}_u(t) = \text{GRU}(\mathbf{s}_u(t^-), \mathbf{m}_u(t))$$

4. **Embedding computation:** Compute node embeddings using temporal graph attention over the temporal neighborhood:
$$\mathbf{z}_u(t) = \text{TemporalAttention}(\mathbf{s}_u(t), \{\mathbf{s}_v(t_v), \Delta t\}_{v \in \mathcal{N}^t(u)})$$

TGN achieves strong performance on dynamic link prediction (predicting future interactions) and dynamic node classification.

---

## 24.10 Molecular Graphs

Molecules have a natural graph structure: atoms are nodes, chemical bonds are edges. Graph neural networks have become the dominant paradigm for molecular property prediction and generation, with transformative applications in drug discovery.

### 24.10.1 Molecular Property Prediction

Given a molecular graph $G$ with atom features (element type, charge, hybridization, aromaticity, etc.) and bond features (bond type, stereochemistry, ring membership), predict properties like toxicity, solubility, binding affinity, or pharmacokinetic parameters.

The pipeline is:
1. **Featurize** atoms and bonds into node/edge feature vectors.
2. **Apply GNN** (GCN, GAT, SchNet, DimeNet, etc.) for message passing.
3. **Readout** (global pooling) to obtain a graph-level representation.
4. **Predict** properties via an MLP head.

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.datasets import MoleculeNet

# Load ESOL (solubility prediction) dataset
dataset = MoleculeNet(root='/tmp/ESOL', name='ESOL')
print(f"Dataset: {len(dataset)} molecules")
print(f"Node features: {dataset.num_node_features}")

class MolecularGNN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim=64, num_layers=3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(num_features, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        # Global mean pooling: aggregate node features to graph level
        x = global_mean_pool(x, batch)
        return self.predictor(x).squeeze(-1)
```

### 24.10.2 Molecule Generation

Generating new molecules with desired properties is a key goal in drug discovery. Approaches include:

- **SMILES-based:** Treat molecules as strings (SMILES notation) and use sequence models (RNNs, Transformers). Simple but may generate chemically invalid strings.
- **Graph-based:** Generate molecular graphs directly, ensuring chemical validity by construction.
- **Fragment-based:** Assemble molecules from chemically meaningful fragments (functional groups, rings).

### 24.10.3 DiffDock: Diffusion for Molecular Docking

Molecular docking predicts how a small molecule (ligand) binds to a protein, a critical step in drug design. Traditional docking methods (AutoDock, Glide) use physics-based scoring functions and sampling.

DiffDock (Corso et al., 2022) formulates docking as a generative diffusion process over the product space of translations, rotations, and torsion angles:

1. **Forward process:** Gradually add noise to the ligand's pose (position, orientation, torsion angles).
2. **Reverse process:** A score model (SE(3)-equivariant graph neural network) predicts the score (gradient of the log-probability) to denoise the pose.
3. **Confidence model:** A separate model ranks the generated poses.

DiffDock significantly outperformed traditional docking methods in blind docking (predicting the binding pose without knowing the binding site), achieving success rates nearly double those of the best physics-based methods.

### 24.10.4 AlphaFold and Structural Biology

While not strictly a graph ML method, AlphaFold (Jumper et al., 2021) revolutionized structural biology by predicting protein 3D structures with near-experimental accuracy. Its architecture includes:

- An *evoformer* module that processes multiple sequence alignments (MSA) and pairwise residue representations using attention mechanisms.
- A structure module that predicts 3D coordinates using equivariant transformations.

AlphaFold's impact extends beyond structure prediction — its representations are used as features for downstream tasks including protein function prediction, drug design, and protein engineering.

---

## 24.11 Graph Generation

Generating realistic graphs is important for drug design, material science, circuit design, and synthetic data generation. The challenge: graphs are discrete, combinatorial objects with variable size and no canonical ordering.

### 24.11.1 Autoregressive Generation: GraphRNN

GraphRNN (You et al., 2018) generates graphs one node at a time. It uses two RNNs:
1. **Graph-level RNN:** Maintains a state summarizing the generated graph so far, generating one node per step.
2. **Edge-level RNN:** For each new node, sequentially generates edges to all preceding nodes.

This autoregressive decomposition:
$$P(G) = \prod_{i=1}^{n} P(\text{node}_i, \text{edges}_i | \text{graph}_{<i})$$

GraphRNN produces realistic graphs (matching statistics like degree distribution, clustering coefficient) but scales poorly with graph size due to the $O(n^2)$ edge decisions.

### 24.11.2 VAE-Based Generation

Graph VAEs encode graphs into a continuous latent space and decode back:

- **Encoder:** GNN produces node embeddings, aggregated to a graph-level latent vector $\mathbf{z}$.
- **Decoder:** Generates the adjacency matrix from $\mathbf{z}$, often factorized as $P(A_{ij} = 1 | \mathbf{z}) = \sigma(\mathbf{z}_i^T \mathbf{z}_j)$.

CGVAE (Conditional Graph VAE) conditions generation on desired properties, enabling property-directed molecule design.

### 24.11.3 Diffusion-Based Graph Generation

Diffusion models for graphs define a forward process that gradually corrupts graph structure (adding/removing edges, noise to features) and learn to reverse it:

- **DiGress (Vignac et al., 2023):** A discrete diffusion model for graphs that operates directly on the adjacency matrix and node features, without requiring continuous relaxations. It uses a graph transformer architecture for the denoising network.
- **GDSS:** Graph Diffusion via the System of SDEs, jointly diffusing over the adjacency matrix and node features.

These approaches produce higher-quality graphs than autoregressive and VAE methods, particularly for molecular generation.

---

## Exercises

### Conceptual Questions

1. **Message passing expressiveness.** Explain the Weisfeiler-Leman test and why it bounds the expressiveness of MPNNs. Give an example of two non-isomorphic graphs that 1-WL (and therefore GCN/GraphSAGE/GAT) cannot distinguish.

2. **GCN vs GAT.** Compare GCN's fixed normalization ($\frac{1}{\sqrt{d_i d_j}}$) with GAT's learned attention coefficients. In what types of graphs would learned attention provide the most benefit?

3. **Over-smoothing analysis.** Mathematically show that repeated application of GCN layers with the normalized adjacency converges to a rank-1 matrix. What does this mean for node distinguishability?

4. **Knowledge graph embeddings.** Prove that TransE cannot model symmetric relations. Show that RotatE can model symmetric relations by setting the relation embedding to be $\mathbf{r} = \pm \mathbf{1}$.

### Implementation Exercises

5. **Custom GNN layer.** Using PyTorch Geometric's `MessagePassing` class, implement GraphSAGE with mean aggregation. Train it on the Cora dataset and compare with GCN.

6. **Knowledge graph completion.** Implement TransE and RotatE for the FB15k-237 dataset. Compute MRR and Hits@10 on the test set. Analyze which relation types each method handles better.

7. **Molecular property prediction.** Train a GNN on the MoleculeNet BBBP dataset (blood-brain barrier permeability). Compare GCN, GAT, and a message-passing network with edge features. Report ROC-AUC with error bars over 5 random seeds.

8. **Temporal graph learning.** Implement a simplified TGN for the Wikipedia dynamic link prediction dataset (available in PyTorch Geometric Temporal). Evaluate using AP (average precision) and compare with a static GNN baseline that ignores temporal information.

### Research Questions

9. **Graph Transformers vs MPNNs.** On the ZINC benchmark (molecular property prediction), compare GPS (MPNN + Transformer) with pure MPNN and pure Transformer baselines. How does combining local and global attention affect performance? What is the computational tradeoff?

10. **Equivariance in molecular GNNs.** Compare SE(3)-equivariant GNNs (SchNet, DimeNet, PaiNN) with standard GNNs on a 3D molecular property prediction task. When does 3D equivariance matter most?

---

## References

1. Bordes, A., Usunier, N., Garcia-Duran, A., Weston, J., & Yakhnenko, O. (2013). Translating Embeddings for Modeling Multi-Relational Data. *NeurIPS*.

2. Brody, S., Alon, U., & Yahav, E. (2022). How Attentive are Graph Attention Networks? *ICLR*.

3. Corso, G., Staerk, H., Jing, B., Barzilay, R., & Jaakkola, T. (2022). DiffDock: Diffusion Steps, Twists, and Turns for Molecular Docking. *ICLR*.

4. Defferrard, M., Bresson, X., & Vandergheynst, P. (2016). Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering. *NeurIPS*.

5. Fey, M., & Lenssen, J. E. (2019). Fast Graph Representation Learning with PyTorch Geometric. *ICLR Workshop on Representation Learning on Graphs and Manifolds*.

6. Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., & Dahl, G. E. (2017). Neural Message Passing for Quantum Chemistry. *ICML*.

7. Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive Representation Learning on Large Graphs. *NeurIPS*.

8. Jumper, J., Evans, R., Pritzel, A., Green, T., Figurnov, M., Ronneberger, O., ... & Hassabis, D. (2021). Highly Accurate Protein Structure Prediction with AlphaFold. *Nature*, 596, 583-589.

9. Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. *ICLR*.

10. Rossi, E., Chamberlain, B., Frasca, F., Eynard, D., Monti, F., & Bronstein, M. (2020). Temporal Graph Networks for Deep Learning on Dynamic Graphs. *ICML Workshop on GRL*.

11. Sun, Z., Deng, Z., Nie, J., & Tang, J. (2019). RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space. *ICLR*.

12. Trouillon, T., Welbl, J., Riedel, S., Gaussier, E., & Bouchard, G. (2016). Complex Embeddings for Simple Link Prediction. *ICML*.

13. Velickovic, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y. (2018). Graph Attention Networks. *ICLR*.

14. Vignac, C., Krawczuk, I., Siraudin, A., Wang, B., Cevher, V., & Frossard, P. (2023). DiGress: Discrete Denoising Diffusion for Graph Generation. *ICLR*.

15. Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). How Powerful are Graph Neural Networks? *ICLR*.

16. Yang, B., Yih, W., He, X., Gao, J., & Deng, L. (2015). Embedding Entities and Relations for Learning and Inference in Knowledge Bases. *ICLR*.

17. Ying, C., Cai, T., Luo, S., Zheng, S., Ke, G., He, D., ... & Liu, T.-Y. (2021). Do Transformers Really Perform Bad for Graph Representation? *NeurIPS*.

18. You, J., Ying, R., Ren, X., Hamilton, W. L., & Leskovec, J. (2018). GraphRNN: Generating Realistic Graphs with an Auto-Regressive Model. *ICML*.

19. Zhang, J., Zhang, H., Xia, C., & Sun, L. (2020). Graph-BERT: Only Attention is Needed for Learning Graph Representations. *arXiv:2001.05140*.
