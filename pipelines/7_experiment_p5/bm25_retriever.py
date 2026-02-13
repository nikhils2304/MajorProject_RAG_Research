from typing import List, Tuple
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
import re

import utils_p5

logger = utils_p5.logger


class BM25Retriever:
    """
    Sparse retriever using BM25 (Okapi) scoring.
    Builds an in-memory index from pre-chunked documents stored in ChromaDB.
    """

    def __init__(self, documents: List[Document]):
        """
        Args:
            documents: List of LangChain Document objects (from ChromaDB collection).
        """
        self.documents = documents
        # Tokenize documents for BM25
        self.tokenized_docs = [self._tokenize(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)
        logger.info(f"BM25 index built with {len(documents)} documents.")

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple whitespace tokenizer with basic cleanup.
        Works across scripts (Latin, Devanagari, Tamil, Bengali) without
        language-specific stemming — keeps the comparison fair.
        """
        # Remove punctuation and lowercase (for Latin chars; no-op for Indic)
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.lower().split()
        # Filter very short tokens
        return [t for t in tokens if len(t) > 1]

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        """
        Retrieve top-k documents with BM25 scores.

        Returns:
            List of (Document, bm25_score) tuples, sorted by score descending.
        """
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            results.append((self.documents[idx], scores[idx]))

        return results
