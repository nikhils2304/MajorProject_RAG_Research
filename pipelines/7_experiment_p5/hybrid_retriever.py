from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma
from bm25_retriever import BM25Retriever

import config
import utils_p5

logger = utils_p5.logger


class HybridRetriever:
    """
    Hybrid retriever that combines Dense (ChromaDB) and Sparse (BM25) retrieval
    using Reciprocal Rank Fusion (RRF).

    RRF formula:
        RRF_score(doc) = 1/(k + rank_dense) + 1/(k + rank_sparse)

    where k is a constant (default 60) that controls how quickly rank importance decays.
    """

    def __init__(self, vector_store: Chroma, bm25_retriever: BM25Retriever, rrf_k: int = 60):
        """
        Args:
            vector_store: ChromaDB vector store for dense retrieval.
            bm25_retriever: BM25Retriever instance for sparse retrieval.
            rrf_k: RRF constant (default 60).
        """
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 10) -> List[Document]:
        """
        Perform hybrid retrieval by fusing dense and sparse results.

        Steps:
            1. Retrieve top_k * 2 from dense (over-retrieve for better fusion)
            2. Retrieve top_k * 2 from BM25
            3. Apply RRF to merge and rank
            4. Return top_k results

        Args:
            query: The search query.
            top_k: Number of final results to return.

        Returns:
            List of LangChain Document objects, ranked by fused RRF score.
        """
        fetch_k = top_k * 2  # Over-retrieve for better fusion coverage

        # 1. Dense retrieval
        dense_results = self.vector_store.similarity_search(query, k=fetch_k)

        # 2. Sparse (BM25) retrieval
        bm25_results = self.bm25_retriever.retrieve(query, top_k=fetch_k)
        bm25_docs = [doc for doc, score in bm25_results]

        # 3. Build RRF scores
        rrf_scores = {}  # doc_id -> (rrf_score, Document)

        # Score dense results
        for rank, doc in enumerate(dense_results):
            doc_id = self._doc_id(doc)
            score = 1.0 / (self.rrf_k + rank + 1)  # rank is 0-indexed, +1 for 1-indexed
            if doc_id in rrf_scores:
                rrf_scores[doc_id] = (rrf_scores[doc_id][0] + score, rrf_scores[doc_id][1])
            else:
                rrf_scores[doc_id] = (score, doc)

        # Score sparse results
        for rank, doc in enumerate(bm25_docs):
            doc_id = self._doc_id(doc)
            score = 1.0 / (self.rrf_k + rank + 1)
            if doc_id in rrf_scores:
                rrf_scores[doc_id] = (rrf_scores[doc_id][0] + score, rrf_scores[doc_id][1])
            else:
                rrf_scores[doc_id] = (score, doc)

        # 4. Sort by fused score and return top_k
        ranked = sorted(rrf_scores.values(), key=lambda x: x[0], reverse=True)
        return [doc for score, doc in ranked[:top_k]]

    def _doc_id(self, doc: Document) -> str:
        """
        Generate a unique identifier for a document.
        Uses source filename + first 100 chars of content to handle duplicate chunks.
        """
        source = doc.metadata.get("source", "unknown")
        content_hash = doc.page_content[:100]
        return f"{source}::{content_hash}"
