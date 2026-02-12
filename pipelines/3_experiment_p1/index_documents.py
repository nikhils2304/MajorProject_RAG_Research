import sys
from pathlib import Path
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent))

import config
import utils_p1
from embeddings import MonolingualEmbedding

logger = utils_p1.logger

def index_language(language: str):
    """
    Indexes documents for a specific language into a dedicated ChromaDB collection.
    """
    logger.info(f"--- Starting Indexing for {language} (Model: {config.MODEL_LABEL}) ---")
    
    # 1. Setup paths
    dataset_folder = config.LANG_TO_FOLDER.get(language)
    if not dataset_folder:
        logger.error(f"No dataset folder mapped for {language}")
        return

    data_path = config.DATASET_DIR / dataset_folder
    persist_dir = config.CHROMA_DB_DIR / f"{language.lower()}_index"
    
    # 2. Check if already indexed
    if persist_dir.exists() and any(persist_dir.iterdir()):
        logger.info(f"Index for {language} already exists at {persist_dir}. Skipping.")
        return

    # 3. Initialize Model
    model_name = config.MODELS[language]
    logger.info(f"Loading Model: {model_name}")
    embedding_fn = MonolingualEmbedding(model_name=model_name)

    # 4. Load Documents
    pdf_files = utils_p1.get_pdf_files(data_path)
    logger.info(f"Found {len(pdf_files)} PDFs in {data_path}")

    documents = []
    for pdf_file in pdf_files:
        text = utils_p1.extract_text_from_pdf(pdf_file, language=language)
        if text:
            # Add metadata for retrieval filtering/verification
            doc = Document(
                page_content=text,
                metadata={"source": pdf_file.name, "language": language}
            )
            documents.append(doc)

    if not documents:
        logger.warning(f"No text extracted for {language}. Skipping.")
        return

    # 5. Chunk Documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents.")

    # 6. Create/Update Vector Store
    # We use a persistent client setting
    vector_store = Chroma(
        collection_name=f"{language.lower()}_collection",
        embedding_function=embedding_fn,
        persist_directory=str(persist_dir)
    )
    
    # Add chunks in batches to avoid hitting limits
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        vector_store.add_documents(batch)
        logger.info(f"Indexed batch {i//batch_size + 1}/{(len(chunks)//batch_size)+1}")

    logger.info(f"Successfully indexed {language} into {persist_dir}")
    logger.info("--------------------------------------------------")

if __name__ == "__main__":
    # Index Indic languages with the active model profile
    for lang in ["Hindi", "Tamil", "Bengali"]:
        index_language(lang)
