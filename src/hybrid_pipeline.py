import time
from src.dense_retriever import DenseRetriever
from src.bm25 import BM25
from src.fusion import reciprocal_rank_fusion

class HybridRAGPipeline:
    """
    This is the main orchestrator class. 
    It strings together our FAISS dense index, our custom BM25 sparse index, 
    and fuses their rankings together using Reciprocal Rank Fusion.
    """
    
    def __init__(self, k1=1.5, b=0.75, rrf_k=60, dense_model_name="all-MiniLM-L6-v2"):
        self.dense_retriever = DenseRetriever(model_name=dense_model_name)
        self.bm25 = None
        self.k1 = k1
        self.b = b
        self.rrf_k = rrf_k
        self.corpus = None

    def build_indexes(self, corpus):
        """
        Build both the FAISS dense index and the BM25 sparse index.
        """
        self.corpus = corpus
        
        # Build Dense Index
        self.dense_retriever.build_index(corpus)
        
        # Build Sparse Index
        print("Building BM25 index from scratch...")
        start = time.time()
        self.bm25 = BM25(corpus, k1=self.k1, b=self.b)
        elapsed = time.time() - start
        print(f"BM25 index built in {elapsed:.2f}s")
        print(f"  Vocabulary size: {len(self.bm25.df):,}")
        print(f"  Avg doc length: {self.bm25.avgdl:.1f} tokens")

    def retrieve(self, query, k=10):
        """
        Retrieve using RRF fusion of dense + BM25.
        
        Fetches k*2 candidates from each retriever, fuses their rankings, 
        and returns the top k.
        """
        if self.bm25 is None or self.dense_retriever.index is None:
            raise ValueError("Indexes not built. Call build_indexes() first.")
            
        # Get top candidates from both retrievers (fetch more than k to ensure coverage)
        dense_ids, dense_scores = self.dense_retriever.retrieve(query, k=k * 2)
        bm25_ids, bm25_scores = self.bm25.retrieve(query, k=k * 2)
        
        # Format as (doc_id, score) pairs
        dense_ranked = list(zip(dense_ids, dense_scores))
        bm25_ranked = list(zip(bm25_ids, bm25_scores))
        
        # Fuse with RRF
        fused = reciprocal_rank_fusion([dense_ranked, bm25_ranked], k=self.rrf_k)
        
        # Return top-k
        top_k_ids = [doc_id for doc_id, _ in fused[:k]]
        top_k_scores = [score for _, score in fused[:k]]
        
        return top_k_ids, top_k_scores
