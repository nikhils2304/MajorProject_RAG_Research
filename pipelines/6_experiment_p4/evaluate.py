
import sys
from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm
from langchain_chroma import Chroma
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer
import time

# Add P4 directory
sys.path.append(str(Path(__file__).parent))
import config
import utils_p4

# Add P1 directory for embeddings
P1_DIR = Path(__file__).parent.parent / "3_experiment_p1"
sys.path.append(str(P1_DIR))
from embeddings import MonolingualEmbedding

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-flash')

logger = utils_p4.logger

def calculate_faithfulness(question, answer, context_text):
    """
    LLM-as-Judge to check if the answer is grounded in context.
    Returns 1.0 for YES, 0.0 for NO.
    """
    start_time = time.time()
    prompt = f"""You are an unbiased evaluator.
    Assess whether the provided Answer is directly supported by the provided Context.
    Ignore outside knowledge.
    
    Context:
    {context_text}
    
    Question: {question}
    Answer: {answer}
    
    Is the Answer faithful to the Context? respond with strictly "YES" or "NO".
    """
    try:
        response = model.generate_content(prompt)
        result = response.text.strip().upper()
        
        # Calculate cost for this call (Evaluator cost)
        # We can track it separately or lump it in. Let's lump it in for Total Experiment Cost.
        # But usually "Cost per 1k queries" refers to the RAG system cost, not evaluation cost.
        # The user likely wants RAG system cost.
        # So I will NOT include this in the main cost metric, but I will log it if needed.
        # For simplicity, I will only track RAG generation cost.
        
        return 1.0 if "YES" in result else 0.0
    except Exception:
        return 0.0

def generate_answer_with_cost(query: str, context_docs: list):
    """Generates an answer using Gemini and tracks cost."""
    context_text = "\n\n".join([doc.page_content for doc in context_docs])
    prompt = f"""You are a helpful assistant. Answer the user's question based strictly on the provided context.
    
    Context:
    {context_text}
    
    Question: {query}
    
    Answer:"""
    
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        
        # Cost Calculation
        input_tokens = response.usage_metadata.prompt_token_count
        output_tokens = response.usage_metadata.candidates_token_count
        
        cost_input = (input_tokens / 1_000_000) * config.GEMINI_INPUT_PRICE_PER_1M
        cost_output = (output_tokens / 1_000_000) * config.GEMINI_OUTPUT_PRICE_PER_1M
        total_cost = cost_input + cost_output
        
        return text, total_cost, context_text
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return "Error generating answer.", 0.0, context_text

def evaluate_language(language: str):
    logger.info(f"--- Starting P4 Evaluation for {language} ---")
    
    # 1. Load QA Dataset
    qa_file = config.QA_DATASET_DIR / config.LANG_TO_QA_FILE[language]
    if not qa_file.exists():
        logger.error(f"QA file not found: {qa_file}")
        return None

    with open(qa_file, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)
    
    # 2. Vector Store
    # 2. Vector Store -> Shared Index (Multilingual Space)
    persist_dir = config.CHROMA_DB_DIR / "shared_index"
    # All languages use the same model in P4
    model_name = config.MODELS["English"] 
    embedding_fn = MonolingualEmbedding(model_name=model_name)
    
    vector_store = Chroma(
        collection_name="shared_multilingual_collection",
        embedding_function=embedding_fn,
        persist_directory=str(persist_dir)
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": config.TOP_K})

    # 3. Loop
    results = []
    total_cost = 0.0
    
    for item in tqdm(qa_pairs, desc=f"Evaluating {language}"):
        query = item.get("question")
        gold_answer = item.get("answer")
        gold_source = item.get("source_file")
        
        # Retrieval
        retrieved_docs = retriever.invoke(query)
        
        # Hit Rate / Recall
        hits = [1 if gold_source in doc.metadata.get("source", "") else 0 for doc in retrieved_docs]
        hit_rate_at_5 = 1 if sum(hits[:5]) > 0 else 0
        hit_rate_at_10 = 1 if sum(hits[:10]) > 0 else 0
        
        # Generation
        generated_answer, cost, context_text = generate_answer_with_cost(query, retrieved_docs[:5])
        total_cost += cost
        
        # Metrics
        # BLEU
        bleu = BLEU()
        bleu_score = bleu.sentence_score(generated_answer, [gold_answer]).score
        
        # ROUGE-L
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_l = scorer.score(gold_answer, generated_answer)['rougeL'].fmeasure
        
        # Faithfulness
        faithfulness = calculate_faithfulness(query, generated_answer, context_text)
        
        results.append({
            "language": language,
            "question": query,
            "hit_rate@5": hit_rate_at_5,
            "hit_rate@10": hit_rate_at_10,
            "bleu": bleu_score,
            "rouge_l": rouge_l,
            "faithfulness": faithfulness,
            "cost": cost,
            "generated_answer": generated_answer,
            "gold_answer": gold_answer
        })

    # 4. Aggregate
    df = pd.DataFrame(results)
    avg_re_5 = df["hit_rate@5"].mean()
    avg_re_10 = df["hit_rate@10"].mean()
    avg_bleu = df["bleu"].mean()
    avg_rouge = df["rouge_l"].mean()
    avg_faith = df["faithfulness"].mean()
    total_sys_cost = df["cost"].sum()
    
    # Cost per 1k queries extrapolation
    cost_per_1k = (total_sys_cost / len(df)) * 1000 if len(df) > 0 else 0.0

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_path = config.OUTPUT_DIR / f"experiment_p4_{language.lower()}_results_{timestamp}.csv"
    df.to_csv(output_path, index=False)
    
    logger.info(f"P4 Evaluation Complete for {language}")
    logger.info(f"Recall@5: {avg_re_5:.4f}")
    logger.info(f"Recall@10: {avg_re_10:.4f}")
    logger.info(f"BLEU: {avg_bleu:.4f}")
    logger.info(f"ROUGE-L: {avg_rouge:.4f}")
    logger.info(f"Faithfulness: {avg_faith:.4f}")
    logger.info(f"Cost per 1k: ${cost_per_1k:.4f}")
    
    return {
        "Language": language,
        "Embedding": config.MODEL_LABEL,
        "Recall@5": avg_re_5,
        "Recall@10": avg_re_10,
        "BLEU": avg_bleu,
        "ROUGE-L": avg_rouge,
        "Faithfulness": avg_faith,
        "Cost_per_1k": cost_per_1k
    }

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run P4 Evaluation")
    parser.add_argument("--language", type=str, help="Specific language to evaluate (e.g. English)")
    args = parser.parse_args()

    languages = [args.language] if args.language else ["English", "Hindi", "Tamil", "Bengali"]
    
    final_metrics = []
    for lang in languages:
        m = evaluate_language(lang)
        if m: final_metrics.append(m)
    
    summary_df = pd.DataFrame(final_metrics)
    print("\n--- Experiment P4 Summary (Multilingual Dense) ---")
    print(summary_df.to_markdown(index=False))
