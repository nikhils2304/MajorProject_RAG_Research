# Multilingual RAG Research (P1-P6)

This repository contains the implementation and evaluation of various Retrieval-Augmented Generation (RAG) strategies for Indian languages (**Hindi, Tamil, Bengali**) alongside **English**.

The project systematically compares:
*   **P1-P3**: Monolingual Baselines (Dense, Hybrid, Reranked).
*   **P4**: Multilingual Dense Retrieval (Shared Vector Space).
*   **P5**: Multilingual Hybrid Retrieval (Shared Dense + BM25 Fusion).
*   **P6**: Multilingual Reranked Retrieval (Cross-Encoder Re-scoring).

## 📂 Project Structure

*   `pipelines/`: Contains the code for each experiment (P1-P6).
*   `docs/`: Contains detailed reports (`PROJECT_REPORT.md`).
*   `Datasets/`: QA pairs and evaluation data.
*   `outputs/`: Logs and result artifacts.

## 🛠️ Setup Guide

### 1. Prerequisites
*   Python 3.10+
*   Git

### 2. Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/nikhils2304/MajorProject_RAG_Research.git
cd MajorProject_RAG_Research
pip install -r requirements.txt
```

### 3. Environment Configuration (Important!)
You must provide a Google Gemini API Key.
Create a `.env` file in the root directory:

```bash
touch .env
```

Add your key to `.env` (Do NOT commit this file):
```
GOOGLE_API_KEY=AIzaSy...YourKeyHere
```

> **Security Note**: This project uses `python-dotenv` to load keys. Never hardcode your API key in the scripts.

---

## 🚀 Running Experiments

### Experiment P4: Multilingual Dense Retrieval
*Strategy: Shared-Space Retrieval using `BAAI/bge-m3`.*

```bash
python pipelines/6_experiment_p4/evaluate.py
```

### Experiment P5: Multilingual Hybrid Retrieval
*Strategy: Dense Retrieval + BM25 (Sparse) fused via RRF.*

```bash
python pipelines/7_experiment_p5/evaluate.py
```

### Experiment P6: Multilingual Reranked Retrieval
*Strategy: Dense Retrieval + Cross-Encoder Reranking (`BAAI/bge-reranker-v2-m3`).*
*(Note: P6 is computationally intensive and requires a GPU).*

```bash
python pipelines/8_experiment_p6/evaluate.py
```

---

## 📊 Results and Analysis

For a detailed breakdown of metrics (Recall, BLEU, ROUGE, Latency) across all experiments, please see:
👉 [**Full Project Report**](docs/PROJECT_REPORT.md)

### quick Summary (P5 Hybrid)
| Language | Recall@10 | BLEU |
| :--- | :--- | :--- |
| **English** | 98.3% | 32.5 |
| **Hindi** | **100.0%** | 42.0 |
| **Bengali** | 97.1% | 28.0 |
| **Tamil** | 88.8% | 17.6 |

---

## Credits
*   **Embedding Model**: `BAAI/bge-m3`
*   **LLM (Judge/Generator)**: Google Gemini 2.0 Flash
*   **Vector DB**: ChromaDB
