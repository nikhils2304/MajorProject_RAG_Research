from typing import List
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings

class MonolingualEmbedding(Embeddings):
    """
    Wrapper for HuggingFace Sentence Transformers to work with LangChain.
    Supports switching models based on initialization.
    """
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        # Clean text
        texts = [t.replace("\n", " ") for t in texts]
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        text = text.replace("\n", " ")
        embedding = self.model.encode(text, show_progress_bar=False)
        return embedding.tolist()
