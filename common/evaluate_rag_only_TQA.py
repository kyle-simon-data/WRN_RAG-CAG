import os
import json
import time
import csv
import psutil
from tqdm import tqdm
from pathlib import Path
from fuzzywuzzy import fuzz
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from rag2.scripts.rag_generate import generate_rag_response

# Paths
TESTDOCS_DIR = Path("data/triviaqa/testdocs")
OUTPUT_CSV = Path("evaluations/evaluation_results_TQA_RAG.csv")
OUTPUT_JSONL = Path("evaluations/evaluation_results_TQA_RAG.jsonl")

# Track completed IDs
completed_ids = set()
if OUTPUT_JSONL.exists():
    with open(OUTPUT_JSONL, "r") as f:
        for line in f:
            try:
                obj = json.loads(line)
                completed_ids.add(obj["question_id"])
            except:
                continue

# BLEU/ROUGE setup
bleu_smoother = SmoothingFunction().method1
rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# RAG answer generator
def generate_rag_answer(question):
    answer, source_set = generate_rag_response(question)
    return answer, list(source_set)

# Answer evaluation
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

# Evaluation loop
file_paths = sorted(TESTDOCS_DIR.glob("*.json"))

with open(OUTPUT_CSV, "a", newline="") as f_csv, open(OUTPUT_JSONL, "a") as f_jsonl:
    writer = None

    for idx, file_path in enumerate(tqdm(file_paths, desc="Evaluating RAG", unit="example")):
        q_id = file_path.stem
        if q_id in completed_ids:
            continue

        with open(file_path, "r") as doc:
            data = json.load(doc)

        question = data.get("question", "")
        ground_truth = data.get("answer", "")

        rag_start = time.time()
        rag_perf_start = psutil.Process().memory_info().rss
        rag_answer, rag_docs = generate_rag_answer(question)
        rag_end = time.time()
        rag_perf_end = psutil.Process().memory_info().rss
        rag_metrics = evaluate_answers(rag_answer, ground_truth)

        row = {
            "question_id": q_id,
            "question": question,
            "ground_truth_answer": ground_truth,
            "rag_answer": rag_answer,
            "rag_top_docs": "; ".join(rag_docs),
            "rag_time_sec": round(rag_end - rag_start, 4),
            "rag_rss_MB_start": round(rag_perf_start / (1024 ** 2), 2),
            "rag_rss_MB_end": round(rag_perf_end / (1024 ** 2), 2),
            "rag_exact_match": rag_metrics["exact_match"],
            "rag_fuzzy_ratio": rag_metrics["fuzzy_ratio"],
            "rag_bleu": rag_metrics["bleu"],
            "rag_rougeL": rag_metrics["rougeL"],
        }

        if writer is None:
            writer = csv.DictWriter(f_csv, fieldnames=row.keys())
            if f_csv.tell() == 0:
                writer.writeheader()

        writer.writerow(row)
        f_csv.flush()
        f_jsonl.write(json.dumps(row) + "\n")
        f_jsonl.flush()

        print(f"[{idx+1}/{len(file_paths)}] Processed {q_id}")

        del rag_answer, rag_docs
        import gc
        gc.collect()
