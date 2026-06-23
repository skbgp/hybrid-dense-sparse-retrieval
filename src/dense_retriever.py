import faiss
import torch
from sentence_transformers import SentenceTransformer

class DenseRetriever:
    """
    A simple wrapper to handle the dense embeddings.
    We use SentenceTransformers to get the vectors, and FAISS for fast cosine similarity lookups.
    """
    
    def __init__(self, model_name="all-MiniLM-L6-v2", device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
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
        corpus_embeddings = self.model.encode(corpus, show_progress_bar=True, batch_size=128)
        corpus_embeddings = corpus_embeddings.astype("float32")
        
        # Build FAISS index
        faiss.normalize_L2(corpus_embeddings)
        self.dimension = corpus_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(corpus_embeddings)
        print(f"FAISS index built: {self.index.ntotal} vectors, {self.dimension} dimensions")

    def retrieve(self, query, k=10):
        """
        Retrieve top-k passages using dense cosine similarity.
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
            
        q_emb = self.model.encode([query]).astype("float32")
        faiss.normalize_L2(q_emb)
        distances, indices = self.index.search(q_emb, k)
        return list(indices[0]), list(distances[0])
