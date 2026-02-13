
import sys
from pathlib import Path
from langchain_chroma import Chroma
import logging
from tqdm import tqdm

# Add config path
sys.path.append(str(Path(__file__).parent))
import config
import utils_p4
from embeddings import MonolingualEmbedding

# Add P1 for embeddings
P1_DIR = Path(__file__).parent.parent / "3_experiment_p1"
sys.path.append(str(P1_DIR))

logger = utils_p4.logger

def merge_indices():
    logger.info("--- Merging Indices into Single Shared Collection ---")
    
    # Target Collection
    shared_persist_dir = config.CHROMA_DB_DIR / "shared_index"
    shared_collection_name = "shared_multilingual_collection"
    
    model_name = config.MODELS["English"] # BAAI/bge-m3 (same for all)
    embedding_fn = MonolingualEmbedding(model_name=model_name)
    
    shared_vector_store = Chroma(
        collection_name=shared_collection_name,
        embedding_function=embedding_fn,
        persist_directory=str(shared_persist_dir)
    )
    
    languages = ["English", "Hindi", "Bengali", "Tamil"]
    
    for lang in languages:
        logger.info(f"Processing {lang}...")
        
        # Source Collection
        source_persist_dir = config.CHROMA_DB_DIR / f"{lang.lower()}_index"
        source_collection_name = f"{lang.lower()}_collection"
        
        if not source_persist_dir.exists():
            logger.warning(f"  Source index for {lang} not found at {source_persist_dir}. Skipping.")
            continue
            
        try:
            # We access ChromaDB directly to get data without re-embedding
            # But LangChain Chroma wrapper doesn't easily expose "get all without embedding".
            # We can use the underlying `_client` and `_collection`.
            
            # Initialize source just to access client
            source_store = Chroma(
                collection_name=source_collection_name,
                embedding_function=embedding_fn,
                persist_directory=str(source_persist_dir)
            )
            
            # Get all data
            # get() returns dict with 'ids', 'embeddings', 'metadatas', 'documents'
            data = source_store.get(include=['embeddings', 'metadatas', 'documents'])
            
            ids = data['ids']
            embeddings = data['embeddings']
            metadatas = data['metadatas']
            documents = data['documents']
            
            if not ids:
                logger.warning(f"  No documents found in {lang}. Skipping.")
                continue
                
            logger.info(f"  Found {len(ids)} documents in {lang}. Merging...")
            
            # Add to Shared Store
            # add_embeddings or add_texts?
            # We have embeddings! So we use `add_embeddings`?
            # Langchain Chroma `add_embeddings` takes text, embedding, metadata.
            # Wait, `Chroma.from_embeddings` exists?
            # `shared_vector_store.add_texts` computes embeddings.
            # We want to PASS existing embeddings.
            
            # Use underlying collection directly to bypass computation
            # shared_vector_store._collection.add(...)
            
            batch_size = 5000
            total = len(ids)
            
            for i in tqdm(range(0, total, batch_size), desc=f"Merging {lang}"):
                batch_ids = ids[i:i+batch_size]
                batch_embeddings = embeddings[i:i+batch_size]
                batch_metadatas = metadatas[i:i+batch_size]
                batch_documents = documents[i:i+batch_size]
                
                # Add "language" to metadata if missing (it should be there)
                for meta in batch_metadatas:
                    meta["original_language"] = lang
                    
                shared_vector_store._collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    documents=batch_documents
                )
                
            logger.info(f"  Merged {lang} successfully.")
            
        except Exception as e:
            logger.error(f"  Failed to merge {lang}: {e}")

    logger.info(f"--- Merge Complete. Shared Index at {shared_persist_dir} ---")

if __name__ == "__main__":
    merge_indices()
