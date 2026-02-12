import sys
from pathlib import Path
import logging

# Add P1 directory to path so we can reuse its utilities
P1_DIR = Path(__file__).parent.parent / "3_experiment_p1"
sys.path.append(str(P1_DIR))  # append so P2's own modules take priority

from utils_p1 import extract_text_from_pdf, get_pdf_files

# Configure Logging for P2
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("experiment_p2.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
