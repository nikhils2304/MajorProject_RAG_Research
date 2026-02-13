
import sys
from pathlib import Path
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Add config path
sys.path.append(str(Path(__file__).parent))

import config
import utils_p4

# Add P1 for embeddings class
P1_DIR = Path(__file__).parent.parent / "3_experiment_p1"
sys.path.append(str(P1_DIR))
from embeddings import MonolingualEmbedding

logger = utils_p4.logger

def index_language(language: str):
    logger.info(f"--- Starting P4 Indexing for {language} ---")
    
    # 1. Setup paths
    dataset_folder = config.LANG_TO_FOLDER.get(language)
    if not dataset_folder:
        logger.error(f"No dataset folder for {language}")
        return

    data_path = config.DATASET_DIR / dataset_folder
    persist_dir = config.CHROMA_DB_DIR / f"{language.lower()}_index"
    
    # Check if exists (SKIP check disabled to force re-indexing for P4 if needed, 
    # but strictly we should check if we want to save time. 
    # Since model changed from P1 (IndicSBERT) to P4 (BGE-M3), we MUST re-index.
    # The dir is new (chroma_db_multilingual), so it should be empty initially.)
    if persist_dir.exists() and any(persist_dir.iterdir()):
        logger.info(f"Index for {language} seems to exist. Skipping to save time. (Delete dir to re-index)")
        return

    # 2. Initialize Model
    model_name = config.MODELS[language]
    logger.info(f"Loading Model: {model_name}")
    embedding_fn = MonolingualEmbedding(model_name=model_name)

    # 3. Load Documents
    pdf_files = utils_p4.get_pdf_files(data_path)
    logger.info(f"Found {len(pdf_files)} PDFs in {data_path}")

    documents = []
    # Reuse extraction logic from utils_p1 (via utils_p4)
    # Note: extraction doesn't depend on model, so it's safe.
    for pdf_file in pdf_files:
        text = utils_p4.extract_text_from_pdf(pdf_file, language=language)
        if text:
            doc = Document(
                page_content=text,
                metadata={"source": pdf_file.name, "language": language}
            )
            documents.append(doc)

    if not documents:
        logger.warning(f"No text extracted for {language}. Skipping.")
        return

    # 4. Chunk
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents.")

    # 5. Create Vector Store
    vector_store = Chroma(
        collection_name=f"{language.lower()}_collection",
        embedding_function=embedding_fn,
        persist_directory=str(persist_dir)
    )
    
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        vector_store.add_documents(batch)
        logger.info(f"Indexed batch {i//batch_size + 1}/{(len(chunks)//batch_size)+1}")

    logger.info(f"Successfully indexed {language} into {persist_dir}")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run P4 Indexing")
    parser.add_argument("--language", type=str, help="Specific language to index (e.g. Bengali)")
    args = parser.parse_args()

    languages = [args.language] if args.language else ["English", "Hindi", "Tamil", "Bengali"]

    for lang in languages:
        index_language(lang)
