import numpy as np
from tqdm import tqdm


def recall_at_k(rankings, gold_labels, k_values):
    """Return the percentage of queries with their gold passage in the top k."""
    if not len(rankings) or len(rankings) != len(gold_labels):
        raise ValueError("Provide one gold passage for each of at least one query.")
    if not len(k_values) or min(k_values) <= 0:
        raise ValueError("Recall cutoffs must be positive.")

    return {
        k: 100 * np.mean([gold in hits[:k] for hits, gold in zip(rankings, gold_labels)])
        for k in k_values
    }


def evaluate_retriever(retrieve_fn, queries, gold_labels, k_values, name=""):
    if not len(k_values) or min(k_values) <= 0:
        raise ValueError("Recall cutoffs must be positive.")
    depth = max(k_values)
    rankings = [
        retrieve_fn(query, k=depth)[0]
        for query in tqdm(queries, desc=f"Evaluating {name}")
    ]
    return recall_at_k(rankings, gold_labels, k_values)


def compute_stats(recalls_list, k):
    """Return the mean and population standard deviation across query samples."""
    values = [recalls[k] for recalls in recalls_list]
    return np.mean(values), np.std(values)
