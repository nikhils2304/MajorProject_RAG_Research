import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Model Configuration
GENERATION_MODEL = "gemini-2.0-flash"
EMBEDDING_MODEL = "text-embedding-004"

# Generation Settings
QA_PAIRS_PER_FILE = 6  # Target number of pairs per file
REQUEST_DELAY = 1.0    # Seconds between API calls

# Dataset Paths (relative to project root)
DATASET_ROOT = "Datasets"
OUTPUT_ROOT = "pipelines/2_eval_dataset"

# Evaluation Thresholds
MIN_QUALITY_SCORE = 4  # Minimum score (out of 5) to keep a QA pair
