import os
import json
import re
import argparse
from collections import defaultdict
import difflib
import ast


# --------------------
# Utilities
# --------------------

def normalize_code(code: str) -> str:
    """Normalize code for fair comparison."""
    return "\n".join(
        line.rstrip()
        for line in code.strip().splitlines()
        if line.strip()
    )


def edit_similarity(a: str, b: str) -> float:
    """Normalized edit similarity (SequenceMatcher)."""
    return difflib.SequenceMatcher(None, a, b).ratio()


def extract_identifiers(code: str):
    """Extract variable / function / class identifiers."""
    identifiers = set()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return identifiers

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            identifiers.add(node.id)
        elif isinstance(node, ast.FunctionDef):
            identifiers.add(node.name)
        elif isinstance(node, ast.ClassDef):
            identifiers.add(node.name)
    return identifiers


def identifier_metrics(pred: str, gt: str):
    P = extract_identifiers(pred)
    G = extract_identifiers(gt)

    em = float(P == G)

    if not P and not G:
        return em, 1.0

    inter = len(P & G)
    precision = inter / len(P) if P else 0.0
    recall = inter / len(G) if G else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0.0
    )
    return em, f1


# --------------------
# Evaluation
# --------------------

def evaluate_repo_completion(
    evaluate_path,
    testset_path,
    prompts_file_path,
    best_prompt_output_path,
):
    # Load testset: task_id -> groundtruth continuation
    test_gt = {}
    with open(testset_path, encoding="utf-8") as f:
        for line in f:
            task = json.loads(line)
            test_gt[task["task_id"]] = task["groundtruth"]

    # scores[prompt_id][metric] = list
    scores = defaultdict(lambda: {
        "code_em": [],
        "code_es": [],
        "id_em": [],
        "id_f1": [],
    })

    # Iterate over completion files (each = one prompt strategy)
    for fname in os.listdir(evaluate_path):
        if not fname.endswith(".jsonl"):
            continue

        m = re.search(r"_(\d+)\.jsonl", fname)
        if not m:
            continue
        prompt_id = int(m.group(1))

        with open(os.path.join(evaluate_path, fname), encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line)
                tid = sample["task_id"]

                if tid not in test_gt:
                    continue

                pred = sample.get("completion", "")
                gt = test_gt[tid]

                pred_n = normalize_code(pred)
                gt_n = normalize_code(gt)

                scores[prompt_id]["code_em"].append(float(pred_n == gt_n))
                scores[prompt_id]["code_es"].append(edit_similarity(pred_n, gt_n))

                idem, idf1 = identifier_metrics(pred, gt)
                scores[prompt_id]["id_em"].append(idem)
                scores[prompt_id]["id_f1"].append(idf1)

    # Aggregate and select best prompt (Identifier F1 is standard)
    best_prompt_id = None
    best_score = -1

    for pid, s in scores.items():
        avg = {k: sum(v) / len(v) for k, v in s.items()}
        print(f"\nPrompt {pid}")
        print(
            f"Code EM: {avg['code_em']:.3f} | "
            f"Code ES: {avg['code_es']:.3f} | "
            f"ID EM: {avg['id_em']:.3f} | "
            f"ID F1: {avg['id_f1']:.3f}"
        )

        if avg["id_f1"] > best_score:
            best_score = avg["id_f1"]
            best_prompt_id = pid

    # Save best prompt
    with open(prompts_file_path, encoding="utf-8") as f_in, \
         open(best_prompt_output_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            p = json.loads(line)
            if p["prompt_id"] == best_prompt_id:
                f_out.write(json.dumps(p) + "\n")

    print(
        f"\nBest prompt_id = {best_prompt_id} "
        f"(Identifier F1 = {best_score:.3f})"
    )


# --------------------
# CLI
# --------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate repo-code completion using CCEval metrics."
    )
    parser.add_argument("--evaluate_path", type=str, required=True)
    parser.add_argument("--testset_path", type=str, required=True)
    parser.add_argument("--origin_prompt", type=str, required=True)
    parser.add_argument("--best_prompt", type=str, required=True)

    args = parser.parse_args()

    evaluate_repo_completion(
        args.evaluate_path,
        args.testset_path,
        args.origin_prompt,
        args.best_prompt,
    )
