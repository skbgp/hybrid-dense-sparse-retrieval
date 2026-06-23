# Hybrid RAG: Lexical & Semantic Retrieval from Scratch

This repository contains an end-to-end Retrieval-Augmented Generation (RAG) pipeline designed to evaluate the trade-offs between dense semantic search and sparse lexical search. 

Instead of relying on standard retrieval libraries, this project implements the BM25 algorithm entirely from scratch (including the inverted index, term frequency saturation, and document length normalization) and fuses it with a FAISS-backed dense retriever using Reciprocal Rank Fusion (RRF).

## Architecture

**Dense Pipeline:** `Query` → `[all-MiniLM-L6-v2]` → `[FAISS Index]` → `Dense Rankings`  
**Sparse Pipeline:** `Query` → `[Custom BM25 Index]` → `Sparse Rankings`  
**Fusion & Generation:** `[Dense + Sparse Rankings]` → `[Reciprocal Rank Fusion]` → `Context` → `[LLM Generator]` → `Final Answer`

## Results & Evaluation

The system was benchmarked on the Google Natural Questions dataset, measuring Recall@5 across multiple random seeds and BM25 hyperparameter combinations.

| Retriever | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|-----------|----------|----------|----------|-----------|
| Dense (FAISS) | 87.2% | 97.5% | 98.7% | 99.5% |
| BM25 (Scratch)| 72.6% | 86.1% | 89.6% | 92.7% |
| Hybrid (RRF) | 83.4% | 93.9% | 96.7% | 98.9% |

Takeaway: While Hybrid RAG is a standard industry practice for out-of-domain retrieval, the ablation study demonstrated that when a dense embedding model (like MiniLM) is already heavily fine-tuned on the target domain, its standalone performance approaches the ceiling (98.7% Recall@5). Fusing it with BM25 introduced lexical noise, lowering the recall score by 2%. This highlights the necessity of running domain-specific ablation tests before deploying hybrid fusion in production.

## Project Structure

```
hybrid-rag-from-scratch/
├── kaggle/
│   └── notebook.py           # self-contained kaggle pipeline
├── src/
│   ├── bm25.py               # custom bm25 implementation
│   └── fusion.py             # reciprocal rank fusion logic
├── requirements.txt
└── README.md
```

## How to Run

The pipeline, including the evaluation loop and ablation sweep, is contained within a single script.

1. Clone this repository.
2. Open `kaggle/notebook.py` in a Jupyter or Kaggle environment.
3. Enable GPU acceleration.
4. Run the script to execute the corpus generation, FAISS indexing, custom BM25 indexing, and the final ablation parameter sweep.

The core BM25 and RRF logic is also abstracted into the `src/` directory for integration into other codebases.
