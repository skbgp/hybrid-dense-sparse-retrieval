"""
Reciprocal Rank Fusion (RRF) for combining multiple ranked lists.

Formula: RRF_score(d) = Σ 1 / (k + rank_i(d))
Default k=60 as per Cormack, Clarke, and Buettcher (2009).
"""

from collections import defaultdict


def reciprocal_rank_fusion(ranked_lists, k=60):
    """
    Args:
        ranked_lists: list of lists, each containing (doc_id, score) tuples
                      sorted by relevance (best first).
        k: smoothing constant.
    Returns:
        List of (doc_id, fused_score) sorted descending.
    """
    if k < 0:
        raise ValueError("The RRF smoothing constant must be non-negative.")
    rrf_scores = defaultdict(float)
    for ranked_list in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked_list):
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


def fuse_results(dense, sparse, k, rrf_k=60):
    """Fuse two (IDs, scores) results and return the top k in the same format."""
    ranked_lists = [zip(*dense), zip(*sparse)]
    top_hits = reciprocal_rank_fusion(ranked_lists, k=rrf_k)[:k]
    return [doc_id for doc_id, _ in top_hits], [score for _, score in top_hits]
