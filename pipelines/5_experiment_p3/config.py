import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATASET_DIR = BASE_DIR / "Datasets"
QA_DATASET_DIR = BASE_DIR / "pipelines/2_eval_dataset"
OUTPUT_DIR = BASE_DIR / "outputs"

# ============================================================
# Model Config — Uses IndicSBERT (primary from P1)
# ============================================================
MODELS = {
    "English": "BAAI/bge-m3",
    "Hindi": "l3cube-pune/indic-sentence-bert-nli",
    "Tamil": "l3cube-pune/indic-sentence-bert-nli",
    "Bengali": "l3cube-pune/indic-sentence-bert-nli",
}

# Reranker Model (multilingual MiniLM, robust and fast)
RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# Reuse P1's ChromaDB indices (no re-indexing needed)
# English was indexed in chroma_db/, Indic languages in chroma_db_indicsbert/
CHROMA_DB_DIR = BASE_DIR / "pipelines/3_experiment_p1/chroma_db_indicsbert"
CHROMA_DB_DIR_ENGLISH = BASE_DIR / "pipelines/3_experiment_p1/chroma_db"


def get_chroma_dir(language: str) -> Path:
    """Returns the correct ChromaDB directory for a given language."""
    if language == "English":
        return CHROMA_DB_DIR_ENGLISH
    return CHROMA_DB_DIR


# Create directories if they don't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Retrieval Settings
TOP_K = 10         # Final top-k after reranking
CANDIDATE_K = 20   # Initial dense retrieval candidates (reduced for speed)

# Dataset Mapping
LANG_TO_FOLDER = {
    "English": "EnglishDataset",
    "Hindi": "HindiDataset",
    "Tamil": "TamilDataset",
    "Bengali": "BengaliDataset",
}

LANG_TO_QA_FILE = {
    "English": "english_qa.json",
    "Hindi": "hindi_qa.json",
    "Tamil": "tamil_qa.json",
    "Bengali": "bengali_qa.json",
}
