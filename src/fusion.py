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
    rrf_scores = defaultdict(float)
    for ranked_list in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked_list):
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
