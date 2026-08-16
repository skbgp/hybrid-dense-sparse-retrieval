# Hybrid RAG from Scratch

I built a retrieval system that combines dense search (FAISS) with sparse search (BM25 written from scratch) using Reciprocal Rank Fusion. The goal was to understand whether hybrid retrieval actually helps, and to implement BM25 manually so I could explain every piece of it.

## How it works

**Dense path:** encode the query with all-MiniLM-L6-v2 into a 384-dim vector, L2-normalize it, and search a FAISS IndexFlatIP for the nearest passages by cosine similarity.

**Sparse path (BM25 from scratch):** build an inverted index over the corpus. For each query term, look up which documents contain it, then score each document using the BM25 formula:
- TF saturation via k1 (default 1.5) -- repeated terms help, but with diminishing returns
- Length normalization via b (default 0.75) -- penalizes long documents so they don't dominate
- IDF weighting -- rare terms matter more than common ones

No `rank_bm25` library. The inverted index, the IDF calculation, the TF saturation -- all written out so I can explain the math.

**Fusion:** take both ranked lists, compute RRF scores as `1/(60 + rank)`, sum across lists, re-sort. Simple but effective -- used in production at Elasticsearch.

**Prompt construction:** the top retrieved passages get formatted into an LLM-ready prompt. The system doesn't call an LLM directly -- it outputs a context-grounded prompt that you can feed to GPT-4, Claude, or any open-source model.

## Evaluation

Evaluated on a 5,000-example subset of Google's Natural Questions dataset. For each query, the system needs to retrieve the correct answer passage from the full corpus. Measured Recall@K across 5 random subsets (seeds 42-46) for statistical reliability.

Three-way ablation: dense-only vs BM25-only vs hybrid (RRF fusion).

There's also a BM25 parameter sensitivity sweep that varies k1 and b to show how they affect both standalone BM25 and hybrid performance.

## What I learned

- **Dense retrieval dominates on NQ.** 98.7% Recall@5 out of the box. When your corpus is 5K passages and your encoder was trained on exactly this type of data, semantic search is hard to beat.
- **BM25 from scratch was worth it.** 89.6% Recall@5 with no model, no GPU, just an inverted index. Writing the scoring formula made me understand why k1 saturates term frequency and why b penalizes long documents. Using a library wouldn't have taught me any of that.
- **Hybrid isn't always better.** Hybrid RRF gets 96.7% Recall@5 -- worse than dense alone. When one retriever is already near the ceiling, mixing in a weaker signal dilutes the ranking. The ablation proves exactly when fusion helps (R@1 for BM25: 72.6 -> Hybrid: 83.4) and when it hurts (R@5 for Dense: 98.7 -> Hybrid: 96.7).
- **BM25 parameters barely matter once you fuse.** The parameter sweep shows BM25 R@5 swings from 86.3 to 89.6 depending on k1/b, but Hybrid R@5 stays locked between 96.6 and 97.0. Dense absorbs the noise.
- **RRF is dead simple and surprisingly effective.** The entire fusion function is 6 lines of code. It doesn't need score calibration between retrievers because it only uses rank positions.

## Structure

```
hybrid-rag-from-scratch/
├── kaggle/
│   └── notebook.py           # runs everything end-to-end
├── src/
│   ├── bm25.py               # BM25 from scratch (inverted index + scoring)
│   ├── dense_retriever.py    # FAISS + MiniLM wrapper
│   ├── fusion.py             # Reciprocal Rank Fusion
│   ├── hybrid_pipeline.py    # orchestrator class tying it all together
│   ├── generation.py         # builds LLM-ready prompts from retrieved passages
│   └── evaluator.py          # Recall@K computation
├── requirements.txt
└── README.md
```

## Running it

### Kaggle (recommended)
1. Create a new Kaggle notebook
2. Enable GPU (Settings -> Accelerator -> GPU T4 x2)
3. First cell: `!pip install sentence-transformers datasets faiss-cpu`
4. Second cell: paste contents of `kaggle/notebook.py`
5. Run

The notebook downloads Natural Questions automatically. No manual dataset setup needed.

### Local
```bash
pip install -r requirements.txt
PYTHONPATH=. python -c "exec(open('kaggle/notebook.py').read())"
```

## Resume bullets

- Implemented BM25 retrieval from scratch (inverted index, IDF smoothing, TF saturation) and combined it with FAISS dense retrieval via Reciprocal Rank Fusion, evaluating on 5K Natural Questions examples across 5 seeds.
- Ran a three-way retrieval ablation (Dense vs BM25 vs Hybrid), showing dense retrieval dominates at 98.7% Recall@5 on NQ while BM25-only achieves 89.6% -- demonstrating that hybrid fusion does not universally improve over strong single retrievers.
- Conducted a BM25 parameter sensitivity sweep over k1 and b, proving that RRF fusion absorbs hyperparameter variance: BM25 R@5 swings 3.3pp across configs while Hybrid R@5 stays within 0.4pp.

