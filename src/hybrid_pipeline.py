import time
from src.dense_retriever import DenseRetriever
from src.bm25 import BM25
from src.fusion import fuse_results

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
        corpus = list(corpus)
        print("Building BM25 index from scratch...")
        start = time.perf_counter()
        bm25 = BM25(corpus, k1=self.k1, b=self.b)
        elapsed = time.perf_counter() - start
        print(f"BM25 index built in {elapsed:.2f}s")
        print(f"  Vocabulary size: {len(bm25.df):,}")
        print(f"  Avg doc length: {bm25.avgdl:.1f} tokens")

        self.dense_retriever.build_index(corpus)
        self.bm25 = bm25
        self.corpus = corpus

    def retrieve(self, query, k=10):
        """
        Retrieve using RRF fusion of dense + BM25.
        
        Fetches k*2 candidates from each retriever, fuses their rankings, 
        and returns the top k.
        """
        if self.bm25 is None or self.dense_retriever.index is None:
            raise ValueError("Indexes not built. Call build_indexes() first.")
            
        dense = self.dense_retriever.retrieve(query, k=k * 2)
        sparse = self.bm25.retrieve(query, k=k * 2)
        return fuse_results(dense, sparse, k, self.rrf_k)
