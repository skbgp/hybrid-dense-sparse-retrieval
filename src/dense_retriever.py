import faiss
import torch
from sentence_transformers import SentenceTransformer

class DenseRetriever:
    """
    A simple wrapper to handle the dense embeddings.
    We use SentenceTransformers to get the vectors, and FAISS for fast cosine similarity lookups.
    """
    
    def __init__(self, model_name="all-MiniLM-L6-v2", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            
        print("Loading sentence embedding model...")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.index = None
        self.dimension = None

    def build_index(self, corpus):
        """
        Pushes all our documents through the embedding model and loads them into FAISS.
        Note: We L2-normalize the vectors so that FAISS Inner Product gives us Cosine Similarity!
        """
        print("Encoding corpus...")
        if len(corpus):
            corpus_embeddings = self.model.encode(
                corpus, show_progress_bar=True, batch_size=128
            ).astype("float32")
            faiss.normalize_L2(corpus_embeddings)
            dimension = corpus_embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(corpus_embeddings)
        else:
            dimension = self.model.get_sentence_embedding_dimension()
            index = faiss.IndexFlatIP(dimension)
        self.index = index
        self.dimension = dimension
        print(f"FAISS index built: {self.index.ntotal} vectors, {self.dimension} dimensions")

    def retrieve(self, query, k=10):
        """
        Retrieve top-k passages using dense cosine similarity.
        """
        return self.retrieve_many([query], k=k)[0]

    def retrieve_many(self, queries, k=10):
        """Encode a batch of queries and return one (IDs, scores) pair per query."""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        if k < 0:
            raise ValueError("k must be non-negative.")
        k = min(k, self.index.ntotal)
        if not len(queries) or k == 0:
            return [([], []) for _ in queries]

        embeddings = self.model.encode(queries, batch_size=128).astype("float32")
        faiss.normalize_L2(embeddings)
        distances, indices = self.index.search(embeddings, k)
        return [(list(ids), list(scores)) for ids, scores in zip(indices, distances)]
