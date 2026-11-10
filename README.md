# Hybrid Dense-Sparse Retrieval for Open-Domain Question Answering

I built this project to see what happens when you combine a lexical retriever and a dense retriever. The sparse path implements BM25 from scratch with an inverted index, smoothed IDF, term-frequency saturation, and document-length normalization. The dense path uses all-MiniLM-L6-v2 embeddings and a FAISS FlatIP index over normalized passage vectors.

Reciprocal Rank Fusion (RRF) combines both ranked lists by adding `1 / (60 + rank)` for each hit, with ranks starting at 1. The pipeline fetches twice the requested number of hits from each retriever before fusing them. `src/generation.py` can format the top passages into a prompt for a separate language model.

## Benchmark

The corpus is the set of unique answer passages from the first 5,000 pairs in the Natural Questions training split. For each of five seeds (42–46), the benchmark samples 1,000 queries without replacement and checks whether each query's paired passage appears in the top results. Those samples can overlap. MiniLM's training data includes Natural Questions, so the numbers describe this retrieval setup and may benefit from training overlap. [Dataset](https://huggingface.co/datasets/sentence-transformers/natural-questions) · [MiniLM training data](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2#training-data)

| Retriever | Recall@1 | Recall@5 | Recall@10 |
| --- | ---: | ---: | ---: |
| Dense | 87.24% | 98.74% | 99.48% |
| BM25 | 72.70% | 89.60% | 92.70% |
| Hybrid | 83.40% | 96.72% | 98.92% |

The BM25 sweep varied `k1` and `b` over five settings. BM25 Recall@5 ranged from 86.30% to 89.60%. Hybrid Recall@5 ranged from 96.60% to 96.98%. RRF improved on BM25 alone, while dense retrieval had the highest recall at every measured cutoff.

## Run it

Use Python 3.10 or newer and run the commands from the repository root.

```bash
python -m pip install -r requirements.txt
python -m src.benchmark
```

For a quick run, pass `--examples 128 --sample-size 32`. The first run downloads the model and dataset. On macOS the benchmark uses one OpenMP thread to avoid a conflict between the PyTorch and FAISS wheels. When using the retrieval classes directly on macOS, start a fresh Python process with `OMP_NUM_THREADS=1` and import the retrieval module before running PyTorch operations. [Upstream issue](https://github.com/pytorch/pytorch/issues/149201)
