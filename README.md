# Hybrid RAG from Scratch

I built a retrieval-augmented generation pipeline to test whether combining BM25 with dense search actually helps. Short answer: it depends on the domain.

I wrote BM25 from scratch (inverted index, TF saturation, length normalization -- the whole thing) and paired it with a FAISS dense retriever using all-MiniLM-L6-v2. The two ranked lists get fused with Reciprocal Rank Fusion, and flan-t5 generates an answer from the top passages.

## How it works

**Dense path:** encode the query with MiniLM into a 384-dim vector, search a FAISS index for the nearest passages by inner product.

**BM25 path:** look up each query word in the inverted index, score documents using the BM25 formula (TF with saturation via k1, length normalization via b, IDF weighting).

**Fusion:** take both ranked lists, compute RRF scores as 1/(60 + rank), sum across lists, re-sort.

**Generation:** concatenate the top passages into a prompt, feed to flan-t5-base, get a grounded answer.

## Results

### In-domain: Natural Questions

| Method | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|--------|----------|----------|----------|-----------|
| Dense (FAISS) | 87.2% | 97.5% | 98.7% | 99.5% |
| BM25 (scratch) | 72.6% | 86.1% | 89.6% | 92.7% |
| Hybrid (RRF) | 83.4% | 93.9% | 96.7% | 98.9% |

Hybrid hurt by ~2pp at Recall@5. Dense was already near the ceiling on this data.

### Out-of-domain: SciFact

| Method | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|--------|----------|----------|----------|-----------|
| Dense (FAISS) | 48.7% | 66.3% | 74.3% | 78.7% |
| BM25 (scratch) | 52.7% | 67.7% | 71.7% | 78.0% |
| Hybrid (RRF) | 51.3% | 70.7% | 77.0% | 84.3% |

Hybrid helped by +2.7pp at Recall@5. BM25 actually beat dense at Recall@1 here -- exact term matching on scientific jargon matters when the dense model hasn't seen the domain.

## What I found

- **Hybrid isn't always better.** On Natural Questions, dense was already at 98.7% Recall@5. Adding BM25 mostly injected lexical noise that pulled good results down. On SciFact, MiniLM hadn't seen scientific text, so BM25's exact matching filled the gap.
- **The finding is two-sided, which is the point.** Most people assume hybrid is always better. Having numbers for both cases is more useful than just showing it works on one dataset.
- **BM25 from scratch was worth it.** Writing the inverted index and the scoring formula made me understand why k1 saturates term frequency, why b penalizes long documents, and how IDF smoothing prevents division by zero. Using a library wouldn't have given me that.
- **Generation is the weakest part.** flan-t5 generates answers but I only have a token-overlap F1 sanity check for it. Proper eval would need faithfulness scoring (is the answer supported by the passages?) using something like RAGAS or an LLM judge.

## Structure

```
hybrid-rag-from-scratch/
├── kaggle/
│   └── notebook.py           # runs everything end-to-end
├── src/
│   ├── bm25.py               # BM25 from scratch (inverted index + scoring)
│   ├── dense_retriever.py    # FAISS + MiniLM wrapper
│   ├── fusion.py             # Reciprocal Rank Fusion
│   ├── hybrid_pipeline.py    # ties it all together
│   ├── generation.py         # flan-t5 answer generation
│   ├── evaluator.py          # Recall@K computation
│   └── eval_ood.py           # SciFact out-of-domain evaluation
├── requirements.txt
└── README.md
```

## Running it

### Kaggle (recommended)
1. Upload `kaggle/notebook.py` to a Kaggle notebook
2. Enable GPU (Settings -> Accelerator -> GPU T4 x2)
3. Run top-to-bottom

The notebook downloads Natural Questions and SciFact, builds both indexes, runs the full ablation, and generates answers.

### Local
```bash
pip install -r requirements.txt
PYTHONPATH=. python -c "exec(open('kaggle/notebook.py').read())"
```

The BM25 and fusion code in `src/` is self-contained if you want to use it in another project.
