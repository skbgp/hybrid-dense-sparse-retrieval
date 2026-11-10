def build_rag_prompt(query, retrieved_docs, corpus, max_chunks=3):
    """
    Takes the top hits from our retriever and formats them into a clean prompt.
    In a real app, you'd just pass this string directly to the OpenAI or Claude API.
    """
    if max_chunks < 0:
        raise ValueError("max_chunks must be non-negative.")
    hits = retrieved_docs[:max_chunks]
    if any(doc_id < 0 or doc_id >= len(corpus) for doc_id in hits):
        raise ValueError("Retrieved passage IDs must point into the corpus.")
    context = "\n---\n".join(
        f"Passage {i}: {corpus[doc_id]}" for i, doc_id in enumerate(hits, start=1)
    )
    prompt = f"""Answer the question based only on the context below.

Context:
{context}

Question: {query}
Answer:"""
    return prompt
