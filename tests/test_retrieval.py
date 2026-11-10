import unittest
from unittest.mock import patch

import numpy as np

from src.bm25 import BM25
from src.generation import build_rag_prompt
from src.hybrid_pipeline import HybridRAGPipeline


class TinyEncoder:
    def encode(self, texts, **kwargs):
        return np.array([[1, 0] if text == "apple" else [0, 1] for text in texts])

    def get_sentence_embedding_dimension(self):
        return 2


class RetrievalTest(unittest.TestCase):
    def test_bm25_leaves_out_zero_score_passages(self):
        retriever = BM25(["apple", "banana"])
        self.assertEqual(retriever.retrieve("unseen"), ([], []))
        self.assertEqual(retriever.retrieve("apple", k=10)[0], [0])
        self.assertEqual(BM25([]).retrieve("apple"), ([], []))

    @patch("src.dense_retriever.SentenceTransformer", return_value=TinyEncoder())
    def test_hybrid_returns_only_valid_passage_ids(self, _):
        pipeline = HybridRAGPipeline()
        pipeline.build_indexes(["apple", "banana"])
        hits, _ = pipeline.retrieve("apple", k=10)
        self.assertEqual(hits, [0, 1])
        self.assertIn("Passage 1: apple", build_rag_prompt("apple", hits, pipeline.corpus))
        self.assertEqual(pipeline.retrieve("apple", k=0), ([], []))
        pipeline.build_indexes([])
        self.assertEqual(pipeline.retrieve("apple"), ([], []))
