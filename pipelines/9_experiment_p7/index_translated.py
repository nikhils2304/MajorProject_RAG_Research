
import sys
from pathlib import Path
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Add config path
sys.path.append(str(Path(__file__).parent))

import config
import utils_p7

# Add P1 for embeddings
P1_DIR = Path(__file__).parent.parent / "3_experiment_p1"
sys.path.append(str(P1_DIR))
from embeddings import MonolingualEmbedding

logger = utils_p7.setup_logger("P7_Indexer", config.PIPELINE_DIR / "indexing.log")

def index_translated_docs():
    logger.info("--- Starting P7 Indexing (English Pivot) ---")
    
    persist_dir = config.CHROMA_DB_DIR
    
    # 1. Initialize Model (English BGE-M3)
    logger.info(f"Loading Model: {config.EMBEDDING_MODEL}")
    embedding_fn = MonolingualEmbedding(model_name=config.EMBEDDING_MODEL)

    # 2. Load Documents from Translated Folder
    documents = []
    
    for language in ["English", "Hindi", "Tamil", "Bengali"]:
        lang_dir = config.PIPELINE_DIR / "translated_docs" / language
        if not lang_dir.exists():
            logger.warning(f"Translated docs for {language} not found. Run translate_docs.py first.")
            continue
            
        txt_files = sorted(list(lang_dir.glob("*.txt")))
        logger.info(f"Found {len(txt_files)} translated docs for {language}")
        
        for txt_file in txt_files:
            try:
                with open(txt_file, "r", encoding="utf-8") as f:
                    text = f.read()
                
                if text:
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": txt_file.stem + ".pdf", # Match original PDF name
                            "original_language": language,
                            "language": "English" # All are English now
                        }
                    )
                    documents.append(doc)
            except Exception as e:
                logger.error(f"Error reading {txt_file}: {e}")

    if not documents:
        logger.error("No documents to index.")
        return

    # 3. Chunk
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents.")

    # 4. Create Vector Store
    # Check if exists and delete? (Optional, but safer for re-runs)
    # Using a new collection name "english_pivot_collection"
    
    vector_store = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embedding_fn,
        persist_directory=str(persist_dir)
    )
    
    batch_size = 5 # Reduced to 5 for maximum stability on CPU
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        vector_store.add_documents(batch)
        logger.info(f"Indexed batch {i//batch_size + 1}/{(len(chunks)//batch_size)+1}")

    logger.info(f"Successfully indexed all docs into {persist_dir}")

if __name__ == "__main__":
    index_translated_docs()
