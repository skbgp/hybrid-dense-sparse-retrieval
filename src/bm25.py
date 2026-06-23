"""
BM25 ranking implemented from scratch.

Contains the full BM25 class with:
- Inverted index construction
- IDF calculation
- TF saturation + length normalisation
- Scoring and top‑k retrieval
"""

import math
import re
import numpy as np
from collections import Counter, defaultdict


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        """
        Args:
            corpus: list of document strings.
            k1: term frequency saturation parameter (1.2-2.0 typical).
            b: length normalisation parameter (0 = no norm, 1 = full norm).
        """
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.tokenized_corpus = [self._tokenize(doc) for doc in corpus]
        doc_lengths = [len(doc) for doc in self.tokenized_corpus]
        self.avgdl = sum(doc_lengths) / len(doc_lengths)
        self.doc_lengths = doc_lengths

        # Inverted index: term -> [(doc_id, tf), ...]
        self.inverted_index = defaultdict(list)
        self.df = Counter()
        for doc_id, tokens in enumerate(self.tokenized_corpus):
            term_freqs = Counter(tokens)
            for term, freq in term_freqs.items():
                self.inverted_index[term].append((doc_id, freq))
                self.df[term] += 1

        # Precompute IDF
        self.idf = {}
        for term, df in self.df.items():
            self.idf[term] = math.log(
                (self.corpus_size - df + 0.5) / (df + 0.5) + 1
            )

    def _tokenize(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()

    def score(self, query):
        query_tokens = self._tokenize(query)
        scores = np.zeros(self.corpus_size, dtype=np.float32)
        for term in query_tokens:
            if term not in self.inverted_index:
                continue
            idf = self.idf[term]
            for doc_id, tf in self.inverted_index[term]:
                dl = self.doc_lengths[doc_id]
                tf_component = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                )
                scores[doc_id] += idf * tf_component
        return scores

    def retrieve(self, query, k=10):
        scores = self.score(query)
        top_indices = np.argsort(scores)[::-1][:k]
        return list(top_indices), list(scores[top_indices])
