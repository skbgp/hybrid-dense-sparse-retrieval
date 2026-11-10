import argparse
import os
import sys

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from src.bm25 import BM25
from src.evaluator import compute_stats, recall_at_k
from src.fusion import fuse_results


SEEDS = (42, 43, 44, 45, 46)
CUTOFFS = (1, 3, 5, 10)
BM25_SETTINGS = ((1.5, 0.75), (0.5, 0.75), (2.0, 0.75), (1.5, 0.0), (1.5, 1.0))


def load_corpus(max_examples):
    rows = load_dataset(
        "sentence-transformers/natural-questions", "pair", split=f"train[:{max_examples}]"
    )
    passage_ids = {}
    queries = []
    gold = []
    for row in rows:
        queries.append(row["query"])
        gold.append(passage_ids.setdefault(row["answer"], len(passage_ids)))
    return list(passage_ids), queries, gold


def benchmark(max_examples, sample_size):
    from src.hybrid_pipeline import HybridRAGPipeline

    corpus, queries, gold = load_corpus(max_examples)
    pipeline = HybridRAGPipeline()
    pipeline.build_indexes(corpus)

    sample_size = min(sample_size, len(queries))
    samples = [
        np.random.RandomState(seed).choice(len(queries), sample_size, replace=False)
        for seed in SEEDS
    ]
    print(f"{len(corpus):,} unique passages from {len(queries):,} pairs")
    print(f"Evaluating {sample_size:,} queries per seed ({', '.join(map(str, SEEDS))})")

    def score_samples(rankings):
        return [
            recall_at_k([rankings[i] for i in sample], [gold[i] for i in sample], CUTOFFS)
            for sample in samples
        ]

    def format_score(scores, k):
        mean, std = compute_stats(scores, k)
        return f"{mean:.2f} ± {std:.2f}"

    depth = max(CUTOFFS)
    dense = pipeline.dense_retriever.retrieve_many(queries, k=2 * depth)
    dense_scores = score_samples([ids for ids, _ in dense])
    sweep = []
    for k1, b in BM25_SETTINGS:
        bm25 = pipeline.bm25 if (k1, b) == BM25_SETTINGS[0] else BM25(corpus, k1=k1, b=b)
        sparse_ids = []
        hybrid_ids = []
        for query, dense_hits in tqdm(zip(queries, dense), total=len(queries), desc=f"k1={k1}, b={b}"):
            sparse_hits = bm25.retrieve(query, k=2 * depth)
            sparse_ids.append(sparse_hits[0])
            hybrid_ids.append(fuse_results(dense_hits, sparse_hits, depth, pipeline.rrf_k)[0])
        sweep.append((k1, b, score_samples(sparse_ids), score_samples(hybrid_ids)))

    print("\nMean Recall@K (%) ± standard deviation")
    print("Retriever  " + "  ".join(f"R@{k:<11}" for k in CUTOFFS))
    for label, scores in (("Dense", dense_scores), ("BM25", sweep[0][2]), ("Hybrid", sweep[0][3])):
        print(f"{label:<10} " + "  ".join(f"{format_score(scores, k):<13}" for k in CUTOFFS))

    print("\nBM25 sensitivity (Recall@5, %)")
    print("k1   b     BM25           Hybrid")
    for k1, b, sparse_scores, hybrid_scores in sweep:
        print(f"{k1:<4} {b:<5} {format_score(sparse_scores, 5):<14} {format_score(hybrid_scores, 5)}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark dense, BM25, and RRF retrieval on Natural Questions.")
    parser.add_argument("--examples", type=int, default=5000, help="Number of question-passage pairs to load.")
    parser.add_argument("--sample-size", type=int, default=1000, help="Queries evaluated per seed.")
    args = parser.parse_args()
    if args.examples < 1 or args.sample_size < 1:
        parser.error("--examples and --sample-size must be positive.")
    if sys.platform == "darwin":
        # The PyTorch and FAISS wheels can load separate OpenMP runtimes.
        os.environ["OMP_NUM_THREADS"] = "1"
    benchmark(args.examples, args.sample_size)


if __name__ == "__main__":
    main()
