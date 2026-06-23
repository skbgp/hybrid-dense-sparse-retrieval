def build_rag_prompt(query, retrieved_docs, corpus, max_chunks=3):
    """
    Takes the top hits from our retriever and formats them into a clean prompt.
    In a real app, you'd just pass this string directly to the OpenAI or Claude API.
    """
    context = "\n---\n".join([f"Passage {i+1}: {corpus[doc_id]}"
                              for i, doc_id in enumerate(retrieved_docs[:max_chunks])])
    prompt = f"""Answer the question based only on the context below.

Context:
{context}

Question: {query}
Answer:"""
    return prompt
