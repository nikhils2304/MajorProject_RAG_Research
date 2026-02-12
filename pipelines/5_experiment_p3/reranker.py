from typing import List
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

import utils_p3

logger = utils_p3.logger


class Reranker:
    """
    Cross-encoder reranker that rescores (query, document) pairs
    using full cross-attention for more accurate relevance scoring.
    """

    def __init__(self, model_name: str):
        """
        Args:
            model_name: HuggingFace model ID for the cross-encoder (e.g. BAAI/bge-reranker-v2-m3).
        """
        logger.info(f"Loading reranker model: {model_name}")
        self.model = CrossEncoder(model_name, trust_remote_code=False, device='cpu')
        self.model_name = model_name
        logger.info("Reranker model loaded.")

    def rerank(self, query: str, documents: List[Document], top_k: int = 10) -> List[Document]:
        """
        Rerank a list of candidate documents for a given query.

        Args:
            query: The search query.
            documents: Candidate documents from initial dense retrieval.
            top_k: Number of top results to return after reranking.

        Returns:
            List of Document objects reranked by cross-encoder relevance score.
        """
        if not documents:
            return []

        # Create (query, document) pairs for cross-encoder scoring
        pairs = [[query, doc.page_content] for doc in documents]

        # Get cross-encoder scores (batch_size=10 for better MPS performance)
        scores = self.model.predict(pairs, batch_size=10)

        # Sort by score descending and return top_k
        scored_docs = list(zip(scores, documents))
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        return [doc for score, doc in scored_docs[:top_k]]
