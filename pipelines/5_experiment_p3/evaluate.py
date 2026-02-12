import sys
from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm
from langchain_chroma import Chroma
import google.generativeai as genai
import os
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer

# Add P3 directory FIRST (so our config.py is found before P1's)
P3_DIR = str(Path(__file__).parent)
P1_DIR = str(Path(__file__).parent.parent / "3_experiment_p1")
if P3_DIR not in sys.path:
    sys.path.insert(0, P3_DIR)
if P1_DIR not in sys.path:
    sys.path.append(P1_DIR)  # append so P3's config takes priority

import config
import utils_p3
from embeddings import MonolingualEmbedding  # Reuse from P1
from reranker import Reranker

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-flash')

logger = utils_p3.logger


def generate_answer(query: str, context_docs: list) -> str:
    """Generates an answer using Gemini given retrieved context."""
    context_text = "\n\n".join([doc.page_content for doc in context_docs])
    prompt = f"""You are a helpful assistant. Answer the user's question based strictly on the provided context.
    
    Context:
    {context_text}
    
    Question: {query}
    
    Answer:"""

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return "Error generating answer."


def evaluate_language(language: str, reranker_instance: Reranker):
    """
    Runs the Reranked RAG evaluation loop for a given language.
    Stage 1: Dense retrieval (over-retrieve CANDIDATE_K docs)
    Stage 2: Cross-encoder reranking to TOP_K
    """
    logger.info(f"--- Starting P3 Reranked Evaluation for {language} ---")
    logger.info(f"  Embedding: {config.MODELS[language]}")
    logger.info(f"  Reranker:  {config.RERANKER_MODEL}")

    # 1. Load QA Dataset
    qa_file = config.QA_DATASET_DIR / config.LANG_TO_QA_FILE[language]
    if not qa_file.exists():
        logger.error(f"QA file not found: {qa_file}")
        return

    with open(qa_file, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)

    logger.info(f"Loaded {len(qa_pairs)} QA pairs from {qa_file}")

    # 2. Initialize Dense Vector Store (Read-Only from P1)
    persist_dir = config.get_chroma_dir(language) / f"{language.lower()}_index"
    model_name = config.MODELS[language]
    embedding_fn = MonolingualEmbedding(model_name=model_name)

    vector_store = Chroma(
        collection_name=f"{language.lower()}_collection",
        embedding_function=embedding_fn,
        persist_directory=str(persist_dir)
    )

    # 3. Evaluation Loop
    results = []

    for item in tqdm(qa_pairs, desc=f"Evaluating {language}"):
        query = item.get("question")
        gold_answer = item.get("answer")
        gold_source = item.get("source_file")

        # Stage 1: Dense retrieval (over-retrieve)
        candidates = vector_store.similarity_search(query, k=config.CANDIDATE_K)

        # Stage 2: Rerank to top_k
        reranked_docs = reranker_instance.rerank(query, candidates, top_k=config.TOP_K)

        # Metrics Calculation
        hits = [1 if gold_source in doc.metadata.get("source", "") else 0 for doc in reranked_docs]
        hit_rate_at_5 = 1 if sum(hits[:5]) > 0 else 0
        hit_rate_at_10 = 1 if sum(hits[:10]) > 0 else 0

        # Generation & Metrics
        generated_answer = generate_answer(query, reranked_docs[:5])

        # BLEU Score
        bleu = BLEU()
        bleu_score = bleu.sentence_score(generated_answer, [gold_answer]).score

        # ROUGE-L Score
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_scores = scorer.score(gold_answer, generated_answer)
        rouge_l = rouge_scores['rougeL'].fmeasure

        results.append({
            "language": language,
            "question": query,
            "gold_source": gold_source,
            "hit_rate@5": hit_rate_at_5,
            "hit_rate@10": hit_rate_at_10,
            "bleu": bleu_score,
            "rouge_l": rouge_l,
            "generated_answer": generated_answer,
            "gold_answer": gold_answer
        })

    # 4. Aggregate Results
    df = pd.DataFrame(results)
    avg_hit_5 = df["hit_rate@5"].mean()
    avg_hit_10 = df["hit_rate@10"].mean()
    avg_bleu = df["bleu"].mean()
    avg_rouge = df["rouge_l"].mean()

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_path = config.OUTPUT_DIR / f"experiment_p3_{language.lower()}_results_{timestamp}.csv"
    df.to_csv(output_path, index=False)

    logger.info(f"P3 Evaluation Complete for {language}")
    logger.info(f"Recall @ 5: {avg_hit_5:.4f}")
    logger.info(f"Recall @ 10: {avg_hit_10:.4f}")
    logger.info(f"Avg BLEU: {avg_bleu:.4f}")
    logger.info(f"Avg ROUGE-L: {avg_rouge:.4f}")
    logger.info(f"Detailed results saved to {output_path}")

    return {
        "Language": language,
        "Experiment": "P3",
        "Embedding": config.MODELS[language],
        "Reranker": config.RERANKER_MODEL,
        "Recall@5": avg_hit_5,
        "Recall@10": avg_hit_10,
        "BLEU": avg_bleu,
        "ROUGE-L": avg_rouge
    }


if __name__ == "__main__":
    # Load reranker once (shared across all languages)
    reranker_instance = Reranker(model_name=config.RERANKER_MODEL)

    final_metrics = []
    for lang in ["English", "Hindi", "Tamil", "Bengali"]:
        metrics = evaluate_language(lang, reranker_instance)
        if metrics:
            final_metrics.append(metrics)

    # Print Summary Table
    summary_df = pd.DataFrame(final_metrics)
    print("\n--- Experiment P3 Reranked Summary ---")
    print(summary_df.to_markdown(index=False))
