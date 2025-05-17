import os
import json
import time
import csv
import psutil
from langchain.schema import Document
from pathlib import Path
from fuzzywuzzy import fuzz
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  #Adds project root to Python path

from cag.cag_pipeline.query_handler_old import load_cache, run_query
from rag.scripts.rag_generate import generate_rag_response


# Paths
TESTDOCS_DIR = Path("data/hotpotqa/testdocs")
OUTPUT_CSV = Path("evaluation_results.csv")

# BLEU/ROUGE setup
bleu_smoother = SmoothingFunction().method1
rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# query functions
def generate_rag_answer(question):
    answer, source_set = generate_rag_response(question)
    return answer, list(source_set)

def generate_cag_answer(question, cache):
    result = run_query(cache, question)
    return result["model_response"], result["context_passages"]

# --- System performance snapshot ---
def get_perf_metrics():
    process = psutil.Process()
    mem_info = process.memory_info()
    cpu_percent = psutil.cpu_percent(interval=None)
    return {
        "rss_memory_MB": mem_info.rss / (1024 * 1024),
        "cpu_percent": cpu_percent,
    }

# --- Answer evaluation ---
def evaluate_answers(generated, ground_truth):
    exact_match = int(generated.strip().lower() == ground_truth.strip().lower())
    fuzzy_ratio = fuzz.ratio(generated, ground_truth)
    reference = [word_tokenize(ground_truth.lower())]
    hypothesis = word_tokenize(generated.lower())
    bleu = sentence_bleu(reference, hypothesis, smoothing_function=bleu_smoother)
    rouge_l = rouge.score(generated, ground_truth)["rougeL"].fmeasure
    return {
        "exact_match": exact_match,
        "fuzzy_ratio": fuzzy_ratio,
        "bleu": bleu,
        "rougeL": rouge_l,
    }

# --- Main evaluation loop ---
results = []

for file_path in sorted(TESTDOCS_DIR.glob("*.json"))[:3]:
    with open(file_path, "r") as f:
        data = json.load(f)

    q_id = file_path.stem
    question = data.get("question", "")
    ground_truth = data.get("answer", "")

    # --- CAG Pipeline ---
    cache = load_cache()

    # Load real context from HotpotQA JSON into cache
    context_paragraphs = []

    for title, paragraph in data.get("context", []):
        # Sometimes paragraph is a list of strings, sometimes it's a string
        if isinstance(paragraph, list):
            context_paragraphs.extend(paragraph)
        else:
            context_paragraphs.append(paragraph)


    wrapped_docs = [Document(page_content=txt) for txt in context_paragraphs]
    cache.add_documents(wrapped_docs)


    cag_start = time.time()
    cag_perf_start = get_perf_metrics()
    cag_answer, cag_docs = generate_cag_answer(question, cache)
    cag_end = time.time()
    cag_perf_end = get_perf_metrics()
    cag_metrics = evaluate_answers(cag_answer, ground_truth)

    # --- RAG Pipeline ---
    rag_start = time.time()
    rag_perf_start = get_perf_metrics()
    rag_answer, rag_docs = generate_rag_answer(question)
    rag_end = time.time()
    rag_perf_end = get_perf_metrics()
    rag_metrics = evaluate_answers(rag_answer, ground_truth)

    # --- Compile Results ---
    results.append({
        "question_id": q_id,
        "question": question,
        "ground_truth_answer": ground_truth,

        "cag_answer": cag_answer,
        "cag_top_docs": "; ".join(cag_docs),
        "cag_time_sec": round(cag_end - cag_start, 4),
        "cag_rss_MB_start": round(cag_perf_start["rss_memory_MB"], 2),
        "cag_rss_MB_end": round(cag_perf_end["rss_memory_MB"], 2),
        "cag_cpu_start": cag_perf_start["cpu_percent"],
        "cag_cpu_end": cag_perf_end["cpu_percent"],
        "cag_exact_match": cag_metrics["exact_match"],
        "cag_fuzzy_ratio": cag_metrics["fuzzy_ratio"],
        "cag_bleu": cag_metrics["bleu"],
        "cag_rougeL": cag_metrics["rougeL"],

        "rag_answer": rag_answer,
        "rag_top_docs": "; ".join(rag_docs),
        "rag_time_sec": round(rag_end - rag_start, 4),
        "rag_rss_MB_start": round(rag_perf_start["rss_memory_MB"], 2),
        "rag_rss_MB_end": round(rag_perf_end["rss_memory_MB"], 2),
        "rag_cpu_start": rag_perf_start["cpu_percent"],
        "rag_cpu_end": rag_perf_end["cpu_percent"],
        "rag_exact_match": rag_metrics["exact_match"],
        "rag_fuzzy_ratio": rag_metrics["fuzzy_ratio"],
        "rag_bleu": rag_metrics["bleu"],
        "rag_rougeL": rag_metrics["rougeL"],
    })


# --- Output to CSV ---
fieldnames = results[0].keys()
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"Evaluation complete. Saved to {OUTPUT_CSV}")
