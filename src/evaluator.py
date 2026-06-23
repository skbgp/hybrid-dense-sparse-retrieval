import numpy as np
from tqdm import tqdm

def evaluate_retriever(retrieve_fn, queries, gold_labels, k_values, name=""):
    """
    Loops through our test queries and checks if the correct 'gold' passage 
    actually shows up in the top K results. Super useful for ablation testing!
    """
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

def compute_stats(recalls_list, k):
    """
    Helper to average out the recall scores over multiple random seeds,
    so we know our results aren't just getting lucky.
    """
    values = [r[k] for r in recalls_list]
    return np.mean(values), np.std(values)
