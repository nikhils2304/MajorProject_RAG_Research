
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

# Add config path
sys.path.append(str(Path(__file__).parent))
import config
import utils_p7

# Add P1 for embeddings
P1_DIR = Path(__file__).parent.parent / "3_experiment_p1"
sys.path.append(str(P1_DIR))
from embeddings import MonolingualEmbedding

load_dotenv()
logger = utils_p7.setup_logger("P7_Evaluator", config.PIPELINE_DIR / "evaluation.log")

# Reuse Gemini model from utils or init new one
model = utils_p7.translation_model

def calculate_faithfulness(question, answer, context_text):
    """
    LLM-as-Judge to check if the answer is grounded in context.
    Returns 1.0 for YES, 0.0 for NO.
    """
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
logger = utils_p7.setup_logger("P7_Evaluator", config.PIPELINE_DIR / "evaluation.log")

def evaluate_language(language: str):
    logger.info(f"--- Starting P7 Pivot Evaluation for {language} ---")
    
    # 1. Load QA Data
    qa_file = config.QA_DATASET_DIR / config.LANG_TO_QA_FILE[language]
    if not qa_file.exists():
        logger.error(f"QA file not found: {qa_file}")
        return

    with open(qa_file, "r") as f:
        qa_pairs = json.load(f)

    # 2. Filter QA Pairs (Top 25 Docs only)
    dataset_folder = config.LANG_TO_PDF_DIR.get(language)
    data_path = config.DATASET_DIR / dataset_folder
    all_pdfs = utils_p7.get_pdf_files(data_path)
    # Match filenames exactly (stems might differ? usually source_file in json has extension)
    # Check one item to be sure. Usually "filename.pdf".
    selected_pdfs_names = {p.name for p in all_pdfs[:config.DOCS_PER_LANG]}
    
    filtered_qa = [qa for qa in qa_pairs if qa.get("source_file") in selected_pdfs_names]
    logger.info(f"Filtered QA Pairs: {len(filtered_qa)} / {len(qa_pairs)} (Top {config.DOCS_PER_LANG} docs)")
    
    if not filtered_qa:
        logger.warning("No QA pairs left after filtering. Skipping.")
        return

    # 3. Load Vector Store (Pivot Index)
    embedding_fn = MonolingualEmbedding(model_name=config.EMBEDDING_MODEL)
    vector_store = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embedding_fn,
        persist_directory=str(config.CHROMA_DB_DIR)
    )
    
    # 4. Evaluation Loop
    results = []
    total_cost = 0.0
    
    for item in tqdm(filtered_qa, desc=f"Evaluating {language}"):
        original_query = item.get("question")
        gold_answer = item.get("answer")
        gold_source = item.get("source_file")
        
        start_time = time.time()
        
        # A. Translate Query -> English
        if language == "English":
            english_query = original_query
        else:
            english_query = utils_p7.translate_text(original_query, target_lang="English")
            
        # B. Retrieve (Dense Search in English)
        try:
            retrieved_docs = vector_store.similarity_search(english_query, k=10)
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            continue

        # Calculate Retrieval Metrics
        hits = [1 if gold_source in doc.metadata.get("source", "") else 0 for doc in retrieved_docs]
        hit_rate_at_5 = 1 if sum(hits[:5]) > 0 else 0
        hit_rate_at_10 = 1 if sum(hits[:10]) > 0 else 0
        
        # C. Generate Answer (English)
        english_answer, cost, context_text = generate_answer_with_cost(english_query, retrieved_docs[:5])
        total_cost += cost
        
        # D. Translate Answer -> Original Language
        if language == "English":
            final_answer = english_answer
        else:
            final_answer = utils_p7.translate_text(english_answer, target_lang=language)
            
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
            "hit_rate@10": hit_rate_at_10,
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
    out_file = config.OUTPUT_DIR / f"experiment_p7_{language.lower()}_results.csv"
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
    print("\n--- P7 Evaluation Summary ---")
    print(f"{'Language':<10} {'R@5':<8} {'R@10':<8} {'BLEU':<8} {'ROUGE':<8} {'Faith':<8} {'Latency':<8} {'Cost/1k':<8}")
    for item in summary:
        print(f"{item['language']:<10} {item['Recall@5']:.4f}   {item['Recall@10']:.4f}   {item['BLEU']:.2f}     {item['ROUGE-L']:.4f}   {item['Faithfulness']:.4f}   {item['Latency']:.0f}ms    ${item['Cost/1k']:.3f}")
