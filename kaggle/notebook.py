# %% [markdown]
# # Hybrid RAG System - Dense + Sparse Retrieval with RRF Fusion
# **BM25 from scratch + FAISS dense search + Reciprocal Rank Fusion**
#
# This notebook builds and evaluates a hybrid retrieval system that combines:
# - **Dense retrieval** (sentence embeddings + FAISS cosine similarity)
# - **Sparse retrieval** (BM25 implemented from scratch - no libraries)
# - **Hybrid fusion** (Reciprocal Rank Fusion to merge both rankings)
#
# We evaluate on **Natural Questions (NQ)** - a real Google QA benchmark - and
# measure Recall@5 for dense-only, BM25-only, and hybrid to prove the fusion helps.
#
# **Runtime:** Enable GPU (`Settings → Accelerator → GPU T4 x2`)

# %% [markdown]
# ## 1. Install Dependencies
# %%
!pip install -q sentence-transformers datasets faiss-cpu

# %% [markdown]
# ## 2. Imports and Setup
# %%
import os
import math
import json
import time
import faiss
import torch
import numpy as np
from collections import Counter, defaultdict
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DATA_DIR = "/kaggle/working/data"
os.makedirs(DATA_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# %% [markdown]
# ## 3. Load Natural Questions Dataset
# We use a subset of Google's Natural Questions from the sentence-transformers hub.
# Each example has a 'query' and an 'answer' passage. We build a corpus from all unique
# answer passages and evaluate whether our retriever can find the correct one.
# %%
print("Loading Natural Questions dataset...")
nq = load_dataset("sentence-transformers/natural-questions", split="train[:5000]")
print(f"Loaded {len(nq)} examples")
print(f"Columns: {nq.column_names}")
print(f"Sample query: {nq[0]['query'][:100]}...")

# Build a corpus of unique passages and a gold-standard mapping
corpus = []
corpus_lookup = {}  # passage text -> corpus index (faster than list.index())
queries = []
gold_labels = []  # For each query, the index of the correct passage in corpus

for example in tqdm(nq, desc="Building corpus"):
    query = example["query"]
    answer = example["answer"]
    
    # Add passage to corpus if not already there
    if answer not in corpus_lookup:
        corpus_lookup[answer] = len(corpus)
        corpus.append(answer)
    
    queries.append(query)
    gold_labels.append(corpus_lookup[answer])

print(f"\nCorpus size: {len(corpus)} unique passages")
print(f"Queries: {len(queries)}")

# %% [markdown]
# ## 4. Dense Retrieval (Sentence Embeddings + FAISS)
# %%
print("Loading sentence embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)

print("Encoding corpus...")
corpus_embeddings = embed_model.encode(corpus, show_progress_bar=True, batch_size=128)
corpus_embeddings = corpus_embeddings.astype("float32")

# Build FAISS index
faiss.normalize_L2(corpus_embeddings)
DIMENSION = corpus_embeddings.shape[1]
index = faiss.IndexFlatIP(DIMENSION)
index.add(corpus_embeddings)
print(f"FAISS index built: {index.ntotal} vectors, {DIMENSION} dimensions")

def dense_retrieve(query, k=10):
    """Retrieve top-k passages using dense cosine similarity."""
    q_emb = embed_model.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb)
    distances, indices = index.search(q_emb, k)
    return list(indices[0]), list(distances[0])

# Quick test
test_indices, test_scores = dense_retrieve(queries[0])
print(f"\nDense test query: '{queries[0][:80]}...'")
print(f"  Top result: '{corpus[test_indices[0]][:80]}...'")

# %% [markdown]
# ## 5. BM25 from Scratch (No Libraries)
# This is the core differentiator. We implement the full BM25 scoring formula manually.
#
# **BM25 Score** = Σ IDF(t) × (tf(t,d) × (k1 + 1)) / (tf(t,d) + k1 × (1 - b + b × |d|/avgdl))
#
# Where:
# - IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
# - tf(t,d) = term frequency of term t in document d
# - k1 = term frequency saturation parameter (controls diminishing returns of repeated terms)
# - b = document length normalization parameter (penalizes long documents)
# - |d| = document length, avgdl = average document length

# %%
# NOTE: BM25 is also available as a standalone module in src/bm25.py for the GitHub repo.
# Here we keep it inline so the Kaggle notebook is fully self-contained.

class BM25:
    """
    BM25 retrieval implemented from scratch.
    
    No rank_bm25 library. Every step is written out so you can
    explain the math in an interview.
    """
    
    def __init__(self, corpus, k1=1.5, b=0.75):
        """
        Build the BM25 index from a list of documents.
        
        Args:
            corpus: list of document strings
            k1: term frequency saturation. Higher = more credit for repeated terms.
                1.2-2.0 is typical. At k1=0, BM25 degenerates to binary matching.
            b: length normalization. b=1.0 fully normalizes by doc length.
                b=0.0 ignores doc length entirely. 0.75 is the standard.
        """
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        
        # Tokenize all documents (simple whitespace + lowercasing)
        self.tokenized_corpus = [self._tokenize(doc) for doc in corpus]
        
        # Compute average document length
        doc_lengths = [len(doc) for doc in self.tokenized_corpus]
        self.avgdl = sum(doc_lengths) / len(doc_lengths)
        self.doc_lengths = doc_lengths
        
        # Build inverted index: term -> list of (doc_id, term_frequency)
        # This is what makes BM25 fast - we only look at documents containing query terms
        self.inverted_index = defaultdict(list)
        self.df = Counter()  # document frequency: how many docs contain each term
        
        for doc_id, tokens in enumerate(self.tokenized_corpus):
            term_freqs = Counter(tokens)
            for term, freq in term_freqs.items():
                self.inverted_index[term].append((doc_id, freq))
                self.df[term] += 1
        
        # Precompute IDF for every term in vocabulary
        # IDF = log((N - df + 0.5) / (df + 0.5) + 1)
        # The +1 inside log ensures IDF is always positive
        self.idf = {}
        for term, df in self.df.items():
            self.idf[term] = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1)
    
    def _tokenize(self, text):
        """Simple whitespace tokenizer with lowercasing and punctuation removal."""
        import re
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()
    
    def score(self, query):
        """
        Score every document against a query.
        
        Returns a numpy array of BM25 scores, one per document.
        """
        query_tokens = self._tokenize(query)
        scores = np.zeros(self.corpus_size, dtype=np.float32)
        
        for term in query_tokens:
            if term not in self.inverted_index:
                continue
            
            idf = self.idf[term]
            
            for doc_id, tf in self.inverted_index[term]:
                # BM25 TF component with saturation
                # As tf grows, this approaches (k1 + 1) asymptotically
                # k1 controls how fast saturation happens
                dl = self.doc_lengths[doc_id]
                tf_component = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
                
                scores[doc_id] += idf * tf_component
        
        return scores
    
    def retrieve(self, query, k=10):
        """Retrieve top-k documents for a query."""
        scores = self.score(query)
        top_k_indices = np.argsort(scores)[::-1][:k]
        top_k_scores = scores[top_k_indices]
        return list(top_k_indices), list(top_k_scores)


# Build BM25 index
print("Building BM25 index from scratch...")
start = time.time()
bm25 = BM25(corpus, k1=1.5, b=0.75)
elapsed = time.time() - start
print(f"BM25 index built in {elapsed:.2f}s")
print(f"  Vocabulary size: {len(bm25.df):,}")
print(f"  Avg doc length: {bm25.avgdl:.1f} tokens")

# Quick test
test_indices_bm25, test_scores_bm25 = bm25.retrieve(queries[0])
print(f"\nBM25 test query: '{queries[0][:80]}...'")
print(f"  Top result: '{corpus[test_indices_bm25[0]][:80]}...'")

# %% [markdown]
# ## 6. Reciprocal Rank Fusion (RRF)
# RRF combines two ranked lists by assigning each document a score based on its rank position.
#
# **RRF Score** = Σ 1 / (k + rank_i)
#
# where k is a constant (typically 60) and rank_i is the document's rank in retriever i.
# This is simple but remarkably effective - used in production at Elasticsearch and others.

# %%
# NOTE: RRF is also available as a standalone module in src/fusion.py for the GitHub repo.
# Here we keep it inline so the Kaggle notebook is fully self-contained.

def reciprocal_rank_fusion(ranked_lists, k=60):
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.
    
    Args:
        ranked_lists: list of lists, each containing (doc_id, score) tuples
                      ordered by relevance (best first)
        k: smoothing constant (default 60, from the original RRF paper)
    
    Returns:
        List of (doc_id, rrf_score) tuples, sorted by fused score descending
    """
    rrf_scores = defaultdict(float)
    
    for ranked_list in ranked_lists:
        for rank, (doc_id, _score) in enumerate(ranked_list):
            # rank is 0-indexed, so rank+1 gives 1-indexed position
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)
    
    # Sort by fused score (highest first)
    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return fused


def hybrid_retrieve(query, k=10):
    """Retrieve using RRF fusion of dense + BM25."""
    # Get top candidates from both retrievers (fetch more than k to ensure coverage)
    dense_ids, dense_scores = dense_retrieve(query, k=k * 2)
    bm25_ids, bm25_scores = bm25.retrieve(query, k=k * 2)
    
    # Format as (doc_id, score) pairs
    dense_ranked = list(zip(dense_ids, dense_scores))
    bm25_ranked = list(zip(bm25_ids, bm25_scores))
    
    # Fuse with RRF
    fused = reciprocal_rank_fusion([dense_ranked, bm25_ranked])
    
    # Return top-k
    top_k_ids = [doc_id for doc_id, _ in fused[:k]]
    top_k_scores = [score for _, score in fused[:k]]
    return top_k_ids, top_k_scores


# Quick test
test_ids_hybrid, test_scores_hybrid = hybrid_retrieve(queries[0])
print(f"Hybrid test query: '{queries[0][:80]}...'")
print(f"  Top result: '{corpus[test_ids_hybrid[0]][:80]}...'")

# %% [markdown]
# ## 7. Evaluation - Dense vs BM25 vs Hybrid Ablation
# This is the Applied Scientist section. We measure Recall@5 for all three retrievers
# and generate the exact resume bullet.

# %%
K_VALUES = [1, 3, 5, 10]

def evaluate_retriever(retrieve_fn, queries, gold_labels, k_values, name=""):
    """Evaluate a retriever on Recall@K."""
    recalls = {k: 0 for k in k_values}
    
    for i, query in enumerate(tqdm(queries, desc=f"Evaluating {name}")):
        retrieved_ids, _ = retrieve_fn(query, k=max(k_values))
        gold = gold_labels[i]
        
        for k in k_values:
            if gold in retrieved_ids[:k]:
                recalls[k] += 1
    
    # Convert counts to percentages
    for k in k_values:
        recalls[k] = (recalls[k] / len(queries)) * 100
    
    return recalls

# Use a subset for evaluation (full set takes too long)
EVAL_SIZE = min(1000, len(queries))
SEEDS = [42, 43, 44, 45, 46]

all_dense_recalls = []
all_bm25_recalls = []
all_hybrid_recalls = []

print(f"Evaluating on {EVAL_SIZE} queries across {len(SEEDS)} random subsets...")

for seed in SEEDS:
    print(f"\n--- Running Seed {seed} ---")
    np.random.seed(seed)
    eval_indices = np.random.choice(len(queries), EVAL_SIZE, replace=False)
    eval_queries = [queries[i] for i in eval_indices]
    eval_labels = [gold_labels[i] for i in eval_indices]
    
    dense_r = evaluate_retriever(dense_retrieve, eval_queries, eval_labels, K_VALUES, "Dense")
    bm25_r = evaluate_retriever(bm25.retrieve, eval_queries, eval_labels, K_VALUES, "BM25")
    hybrid_r = evaluate_retriever(hybrid_retrieve, eval_queries, eval_labels, K_VALUES, "Hybrid")
    
    all_dense_recalls.append(dense_r)
    all_bm25_recalls.append(bm25_r)
    all_hybrid_recalls.append(hybrid_r)

def compute_stats(recalls_list, k):
    values = [r[k] for r in recalls_list]
    return np.mean(values), np.std(values)

# %% [markdown]
# ## 8. Results & Resume Bullet Generator
# %%
print("\n" + "=" * 80)
print(f"RETRIEVAL ABLATION RESULTS (Mean ± Std over {len(SEEDS)} seeds)")
print("=" * 80)
print(f"{'Method':<20} | {'Recall@1':<15} | {'Recall@3':<15} | {'Recall@5':<15} | {'Recall@10':<15}")
print("-" * 80)

final_stats = {}
for name, results_list in [("Dense (FAISS)", all_dense_recalls), 
                           ("BM25 (scratch)", all_bm25_recalls), 
                           ("Hybrid (RRF)", all_hybrid_recalls)]:
    row = f"{name:<20} | "
    final_stats[name] = {}
    for k in K_VALUES:
        mean, std = compute_stats(results_list, k)
        row += f"{mean:>5.1f} ± {std:<5.1f} | "
        final_stats[name][k] = mean
    print(row)

# Compute improvement
dense_r5 = final_stats["Dense (FAISS)"][5]
bm25_r5 = final_stats["BM25 (scratch)"][5]
hybrid_r5 = final_stats["Hybrid (RRF)"][5]
best_single = max(dense_r5, bm25_r5)
improvement = hybrid_r5 - best_single

print(f"\nRESUME BULLET GENERATOR:")
print(f'"Combined dense and sparse retrieval using Reciprocal Rank Fusion (RRF), '
      f'improving Recall@5 by {improvement:.1f} percentage points over the best standalone '
      f'retriever ({hybrid_r5:.1f}% vs {best_single:.1f}%), averaged across {len(SEEDS)} evaluation seeds."')

# %% [markdown]
# ## 9. Ablation: BM25 Parameter Sensitivity (Optional)
# Change k1 and b below and re-run Section 5 and 7 to see how parameters affect retrieval.
# - k1 = 1.5 (default) → try 0.5, 1.0, 2.0
# - b = 0.75 (default) → try 0.0, 0.5, 1.0
#
# | k1  | b    | BM25 Recall@5 | Hybrid Recall@5 |
# |-----|------|---------------|-----------------|
# | 1.5 | 0.75 |               |                 |
# | 0.5 | 0.75 |               |                 |
# | 2.0 | 0.75 |               |                 |
# | 1.5 | 0.0  |               |                 |
# | 1.5 | 1.0  |               |                 |

# %% [markdown]
# ## 9.1. BM25 Parameter Sensitivity Sweep
# %%
# (no extra imports needed - BM25, reciprocal_rank_fusion, evaluate_retriever already defined above)

param_grid = [
    (1.5, 0.75),   # default
    (0.5, 0.75),
    (2.0, 0.75),
    (1.5, 0.0),
    (1.5, 1.0),
]

EVAL_SIZE = 1000
SEEDS = [42, 43, 44, 45, 46]
K_VALUES = [1, 3, 5, 10]

def eval_bm25_params(k1, b, eval_queries, eval_labels):
    # Build a fresh BM25 index with given parameters
    bm25_tmp = BM25(corpus, k1=k1, b=b)
    # Evaluate BM25 only
    bm25_recalls = evaluate_retriever(bm25_tmp.retrieve, eval_queries, eval_labels, K_VALUES, f"BM25 k1={k1} b={b}")
    # Evaluate hybrid (dense + this BM25)
    def hybrid_tmp(query, k=10):
        dense_ids, dense_scores = dense_retrieve(query, k=k*2)
        bm25_ids, bm25_scores = bm25_tmp.retrieve(query, k=k*2)
        fused = reciprocal_rank_fusion([list(zip(dense_ids, dense_scores)),
                                        list(zip(bm25_ids, bm25_scores))])
        top_ids = [doc_id for doc_id, _ in fused[:k]]
        top_scores = [score for _, score in fused[:k]]
        return top_ids, top_scores
    hybrid_recalls = evaluate_retriever(hybrid_tmp, eval_queries, eval_labels, K_VALUES, f"Hybrid k1={k1} b={b}")
    return bm25_recalls, hybrid_recalls

# Pre‑sample evaluation subsets once
eval_subsets = {}
for seed in SEEDS:
    np.random.seed(seed)
    indices = np.random.choice(len(queries), EVAL_SIZE, replace=False)
    eq = [queries[i] for i in indices]
    el = [gold_labels[i] for i in indices]
    eval_subsets[seed] = (eq, el)

# Run sweep
results_table = {}
for k1, b in param_grid:
    bm25_r5_list = []
    hybrid_r5_list = []
    for seed in SEEDS:
        eq, el = eval_subsets[seed]
        bm25_r, hybrid_r = eval_bm25_params(k1, b, eq, el)
        bm25_r5_list.append(bm25_r[5])
        hybrid_r5_list.append(hybrid_r[5])
    results_table[(k1, b)] = {
        'bm25_mean': np.mean(bm25_r5_list), 'bm25_std': np.std(bm25_r5_list),
        'hybrid_mean': np.mean(hybrid_r5_list), 'hybrid_std': np.std(hybrid_r5_list)
    }

# Print results
print("\n" + "=" * 60)
print("BM25 Parameter Sensitivity (Recall@5)")
print("=" * 60)
print(f"{'k1':<6} {'b':<6} {'BM25 R@5':<20} {'Hybrid R@5':<20}")
print("-" * 60)
for (k1, b), vals in results_table.items():
    print(f"{k1:<6.1f} {b:<6.2f} {vals['bm25_mean']:>5.1f} ± {vals['bm25_std']:.1f}       {vals['hybrid_mean']:>5.1f} ± {vals['hybrid_std']:.1f}")

# %% [markdown]
# ## 10. The "G" in RAG - Generation
# %%
def build_rag_prompt(query, retrieved_docs, max_chunks=3):
    """
    Build an LLM‑ready prompt using the top retrieved passages.
    In production you would send this to GPT‑4, Claude, or an open‑source model.
    """
    context = "\n---\n".join([f"Passage {i+1}: {corpus[doc_id]}"
                              for i, doc_id in enumerate(retrieved_docs[:max_chunks])])
    prompt = f"""Answer the question based only on the context below.

Context:
{context}

Question: {query}
Answer:"""
    return prompt

# Test on a sample query
sample_query = queries[0]
hybrid_ids, _ = hybrid_retrieve(sample_query, k=5)
prompt = build_rag_prompt(sample_query, hybrid_ids)
print("=== RAG Prompt ===")
print(prompt[:800])
print("...")
