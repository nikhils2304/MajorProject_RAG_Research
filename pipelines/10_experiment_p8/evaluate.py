
import sys
import json
import time
from pathlib import Path
from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_core.documents import Document
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer
import pandas as pd
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
import numpy as np

# Add config path
sys.path.append(str(Path(__file__).parent))
import config
import utils_p8

# Add P1 for embeddings
P1_DIR = Path(__file__).parent.parent / "3_experiment_p1"
sys.path.append(str(P1_DIR))
from embeddings import MonolingualEmbedding

load_dotenv()
logger = utils_p8.setup_logger("P8_Evaluator", config.PIPELINE_DIR / "evaluation.log")

# Reuse Gemini model from utils
model = utils_p8.translation_model

# --- Helper Functions (Search & Generation) ---

def build_bm25_index(documents):
    """Builds an in-memory BM25 index from a list of LangChain Documents."""
    tokenized_corpus = [doc.page_content.split() for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25

def reciprocal_rank_fusion(results_k_list, k=60):
    """
    Fused ranks from multiple lists of results.
    results_k_list: List of List of Documents.
    Returns: List of (Document, score) sorted by score.
    """
    fused_scores = {}
    
    for docs in results_k_list:
        for rank, doc in enumerate(docs):
            # Using source filename as unique identifier for deduplication
            doc_id = doc.metadata.get("source")
            if not doc_id: continue
            
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {"doc": doc, "score": 0.0}
            
            fused_scores[doc_id]["score"] += 1 / (k + rank + 1)
            
    sorted_results = sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in sorted_results]

def calculate_faithfulness(question, answer, context_text):
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
        return 1.0 if "YES" in result else 0.0
    except Exception:
        return 0.0

def generate_answer_with_cost(query: str, context_docs: list):
    context_text = "\n\n".join([doc.page_content for doc in context_docs])
    prompt = f"""You are a helpful assistant. Answer the user's question based strictly on the provided context.
    
    Context:
    {context_text}
    
    Question: {query}
    
    Answer:"""
    
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        
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
    logger.info(f"--- Starting P8 Hybrid Pivot Evaluation for {language} ---")
    
    # 1. Load QA Data
    qa_file = config.QA_DATASET_DIR / config.LANG_TO_QA_FILE[language]
    if not qa_file.exists():
        logger.error(f"QA file not found: {qa_file}")
        return None

    with open(qa_file, "r") as f:
        qa_pairs = json.load(f)

    # 2. Filter QA Pairs (Top 25 Docs only)
    dataset_folder = config.LANG_TO_PDF_DIR.get(language)
    data_path = config.DATASET_DIR / dataset_folder
    all_pdfs = utils_p8.get_pdf_files(data_path)
    # Be careful with logic here. config.DOCS_PER_LANG is not defined in P8 config 
    # but inherited conceptually? P7 defined it. Let's assume top 25 logic holds.
    # We should filter based on the available translated docs from P7 implicitly.
    
    # Actually, simpler: Use the translated docs we load to build the allow-list.
    
    # 3. Load & Build Indexes
    
    # A. Pivot Corpora (English)
    # Crucial: For P8, we search in the *English* translated corpus for ALL languages.
    # So we need to load ALL translated english docs (from En, Hi, Ta, Bn source).
    
    all_english_docs = []
    for lang_code in ["English", "Hindi", "Tamil", "Bengali"]:
        docs = utils_p8.load_translated_docs(lang_code)
        all_english_docs.extend(docs)
    
    logger.info(f"Loaded {len(all_english_docs)} translated English documents for Hybrid Search.")
    
    if not all_english_docs:
        logger.error("No documents loaded. Aborting.")
        return None

    # Filter QA based on loaded docs
    loaded_sources = {d.metadata["source"] for d in all_english_docs}
    filtered_qa = [qa for qa in qa_pairs if qa.get("source_file") in loaded_sources]
    logger.info(f"Filtered QA Pairs: {len(filtered_qa)} / {len(qa_pairs)}")
    
    # B. Dense Index (Chroma - Existing)
    embedding_fn = MonolingualEmbedding(model_name=config.EMBEDDING_MODEL)
    vector_store = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embedding_fn,
        persist_directory=str(config.CHROMA_DB_DIR)
    )
    
    # C. Sparse Index (BM25 - on-the-fly)
    bm25 = build_bm25_index(all_english_docs)
    # Map bm25 indices back to documents
    bm25_doc_map = all_english_docs 
    
    # 4. Evaluation Loop
    results = []
    
    for item in tqdm(filtered_qa, desc=f"Evaluating {language}"):
        original_query = item.get("question")
        gold_answer = item.get("answer")
        gold_source = item.get("source_file")
        
        start_time = time.time()
        
        # A. Translate Query -> English
        if language == "English":
            english_query = original_query
        else:
            english_query = utils_p8.translate_text(original_query, target_lang="English")
            # Rate limit mitigation
            time.sleep(0.5) 
            
        # B. Hybrid Retrieval (English)
        
        # 1. Dense
        try:
            dense_results = vector_store.similarity_search(english_query, k=config.TOP_K_DENSE)
        except Exception:
            dense_results = []
            
        # 2. Sparse
        tokenized_query = english_query.split()
        if not tokenized_query:
            sparse_results = []
        else:
            # get_top_n returns list of docs
            sparse_results = bm25.get_top_n(tokenized_query, bm25_doc_map, n=config.TOP_K_SPARSE)
            
        # 3. Fusion
        fused_results = reciprocal_rank_fusion([dense_results, sparse_results], k=config.RRF_K)
        final_docs = fused_results[:config.FINAL_TOP_K]
        
        # Calculate Retrieval Metrics
        hits = [1 if gold_source in doc.metadata.get("source", "") else 0 for doc in final_docs]
        hit_rate_at_5 = 1 if sum(hits[:5]) > 0 else 0
        hit_rate_at_10 = 1 if sum(hits[:10]) > 0 else 0 # Actually top_k is 5, so this is same as 5 unless we retrieve more
        # If FINAL_TOP_K is 5, then R@10 is effectively R@5. Let's retrieve 10 for metrics?
        # Standard: usually retrieve top 10 for R@10. Let's adjust fused slice for metrics.
        metrics_docs = fused_results[:10]
        hits_10 = [1 if gold_source in doc.metadata.get("source", "") else 0 for doc in metrics_docs]
        hit_rate_at_10_real = 1 if sum(hits_10) > 0 else 0
        
        # C. Generate Answer (English)
        english_answer, cost, context_text = generate_answer_with_cost(english_query, final_docs)
        
        # D. Translate Answer -> Original Language
        if language == "English":
            final_answer = english_answer
        else:
            final_answer = utils_p8.translate_text(english_answer, target_lang=language)
            time.sleep(0.5)
            
        latency = (time.time() - start_time) * 1000
        
        # E. Metrics
        bleu = BLEU()
        try:
            bleu_score = bleu.sentence_score(final_answer, [gold_answer]).score
        except:
            bleu_score = 0.0
        
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_l = scorer.score(gold_answer, final_answer)['rougeL'].fmeasure
        
        faithfulness = calculate_faithfulness(english_query, english_answer, context_text)
        
        results.append({
            "language": language,
            "hit_rate@5": hit_rate_at_5,
            "hit_rate@10": hit_rate_at_10_real,
            "bleu": bleu_score,
            "rouge_l": rouge_l,
            "faithfulness": faithfulness,
            "cost": cost,
            "latency": latency
        })
        
    # 5. Aggregate
    df = pd.DataFrame(results)
    if df.empty: return None
    
    avg_recall_5 = df["hit_rate@5"].mean()
    avg_recall_10 = df["hit_rate@10"].mean()
    avg_bleu = df["bleu"].mean()
    avg_rouge = df["rouge_l"].mean()
    avg_faithfulness = df["faithfulness"].mean()
    avg_latency = df["latency"].mean()
    total_cost_per_1k = (df["cost"].sum() / len(df)) * 1000
    
    logger.info(f"Results for {language}: R@10={avg_recall_10:.4f}, BLEU={avg_bleu:.2f}, Latency={avg_latency:.0f}ms")
    
    # Save CSV
    out_file = config.OUTPUT_DIR / f"experiment_p8_{language.lower()}_results.csv"
    df.to_csv(out_file, index=False)
    
    return {
        "language": language,
        "Recall@5": avg_recall_5,
        "Recall@10": avg_recall_10,
        "BLEU": avg_bleu,
        "ROUGE-L": avg_rouge,
        "Faithfulness": avg_faithfulness,
        "Latency": avg_latency,
        "Cost/1k": total_cost_per_1k
    }

if __name__ == "__main__":
    summary = []
    for lang in ["English", "Hindi", "Tamil", "Bengali"]:
        res = evaluate_language(lang)
        if res:
            summary.append(res)
            
    # Print Summary Table
    print("\n--- P8 Hybrid Pivot Evaluation Summary ---")
    print(f"{'Language':<10} {'R@5':<8} {'R@10':<8} {'BLEU':<8} {'ROUGE':<8} {'Faith':<8} {'Latency':<8} {'Cost/1k':<8}")
    for item in summary:
        print(f"{item['language']:<10} {item['Recall@5']:.4f}   {item['Recall@10']:.4f}   {item['BLEU']:.2f}     {item['ROUGE-L']:.4f}   {item['Faithfulness']:.4f}   {item['Latency']:.0f}ms    ${item['Cost/1k']:.3f}")
