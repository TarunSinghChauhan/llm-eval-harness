#!/usr/bin/env python3
"""
Small, reproducible eval run:
  1. Samples N MMLU questions (fixed seed).
  2. Gets an answer letter from each model.
  3. Scores exact match with a bootstrapped 95% CI.
  4. Has two judge models label each answer CORRECT/INCORRECT
     WITHOUT seeing the reference, then reports Cohen's kappa
     between the judges and each judge's accuracy vs ground truth.

Works with any OpenAI-compatible API. Defaults to Groq's free tier.

Setup:
    pip install datasets openai numpy
    export GROQ_API_KEY=...        # Windows PowerShell: $env:GROQ_API_KEY="..."
    python eval_run.py

Progress is cached in results/cache.json, so you can stop and re-run
(for example after hitting a rate limit) and it resumes where it left off.
"""
import argparse
import datetime
import json
import os
import random
import re
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from openai import OpenAI

LETTERS = "ABCD"


def load_cache(path):
    return json.loads(path.read_text()) if path.exists() else {}


def save_cache(path, cache):
    path.write_text(json.dumps(cache, indent=1))


def ask(client, model, prompt, delay, retries=5):
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10,
            )
            time.sleep(delay)
            return (r.choices[0].message.content or "").strip()
        except Exception as e:  # rate limits, transient errors
            wait = 5 * (attempt + 1)
            print(f"    retry {attempt + 1}/{retries} in {wait}s: {e}")
            time.sleep(wait)
    return None


def cached(cache, cache_path, key, fn):
    if key not in cache or cache[key] is None:
        cache[key] = fn()
        save_cache(cache_path, cache)
    return cache[key]


def format_question(row):
    opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(row["choices"]))
    return f"Question: {row['question']}\n{opts}"


def parse_letter(text):
    if not text:
        return None
    m = re.search(r"\b([ABCD])\b", text.upper())
    return m.group(1) if m else None


def parse_verdict(text):
    if not text:
        return None
    t = text.upper()
    if "INCORRECT" in t:
        return "INCORRECT"
    if "CORRECT" in t:
        return "CORRECT"
    return None


def bootstrap_ci(values, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    v = np.array(values, dtype=float)
    means = [rng.choice(v, size=len(v), replace=True).mean() for _ in range(n_boot)]
    return {
        "mean": round(float(v.mean()), 4),
        "ci_lower": round(float(np.percentile(means, 2.5)), 4),
        "ci_upper": round(float(np.percentile(means, 97.5)), 4),
        "n": int(len(v)),
    }


def cohen_kappa(a, b):
    a, b = np.array(a), np.array(b)
    po = float((a == b).mean())
    pe = sum(float((a == c).mean()) * float((b == c).mean()) for c in set(a) | set(b))
    if pe >= 1:
        return None  # undefined when both judges always give the same single label
    return round((po - pe) / (1 - pe), 4)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--models", nargs="+",
                   default=["llama-3.3-70b-versatile", "llama-3.1-8b-instant"])
    p.add_argument("--judges", nargs=2,
                   default=["llama-3.3-70b-versatile", "llama-3.1-8b-instant"])
    p.add_argument("--base-url", default="https://api.groq.com/openai/v1")
    p.add_argument("--key-env", default="GROQ_API_KEY")
    p.add_argument("--delay", type=float, default=2.2,
                   help="seconds between calls (Groq free tier is ~30 requests/min)")
    p.add_argument("--out", default="results")
    args = p.parse_args()

    api_key = os.environ.get(args.key_env)
    if not api_key:
        raise SystemExit(f"Set {args.key_env} first.")
    client = OpenAI(base_url=args.base_url, api_key=api_key)

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    cache_path = out / "cache.json"
    cache = load_cache(cache_path)

    ds = load_dataset("cais/mmlu", "all", split="test")
    idxs = random.Random(args.seed).sample(range(len(ds)), args.n)
    rows = [ds[i] for i in idxs]
    truth = [LETTERS[r["answer"]] for r in rows]

    # 1) model answers + exact match
    answers = {m: [] for m in args.models}
    for m in args.models:
        print(f"Answering with {m} ...")
        for j, row in enumerate(rows):
            prompt = ("Answer the multiple-choice question. "
                      "Reply with only the letter (A, B, C, or D).\n\n"
                      + format_question(row) + "\nAnswer:")
            raw = cached(cache, cache_path, f"ans|{m}|{idxs[j]}",
                         lambda: ask(client, m, prompt, args.delay))
            answers[m].append(parse_letter(raw))

    metrics = {}
    for m in args.models:
        em = [1 if a == t else 0 for a, t in zip(answers[m], truth)]
        metrics[m] = {"exact_match": bootstrap_ci(em),
                      "unparseable": sum(a is None for a in answers[m])}

    # 2) judges label each answer without seeing the reference
    verdicts = {j: [] for j in args.judges}
    judged_items = [(m, j_idx) for m in args.models for j_idx in range(len(rows))]
    for judge in args.judges:
        print(f"Judging with {judge} ...")
        for m, j_idx in judged_items:
            letter = answers[m][j_idx]
            prompt = ("Here is a multiple-choice question and a proposed answer. "
                      "Using your own knowledge, decide whether the proposed answer "
                      "is correct. Reply with only CORRECT or INCORRECT.\n\n"
                      + format_question(rows[j_idx])
                      + f"\nProposed answer: {letter}")
            raw = cached(cache, cache_path, f"judge|{judge}|{m}|{idxs[j_idx]}",
                         lambda: ask(client, judge, prompt, args.delay))
            verdicts[judge].append(parse_verdict(raw))

    # keep only items both judges labelled
    valid = [i for i in range(len(judged_items))
             if all(verdicts[j][i] is not None for j in args.judges)]
    j1, j2 = args.judges
    a = [verdicts[j1][i] for i in valid]
    b = [verdicts[j2][i] for i in valid]

    gt = [("CORRECT" if answers[m][ix] == truth[ix] else "INCORRECT")
          for (m, ix) in judged_items]
    judge_accuracy = {
        j: round(float(np.mean([verdicts[j][i] == gt[i] for i in valid])), 4)
        for j in args.judges
    }

    summary = {
        "date": datetime.date.today().isoformat(),
        "dataset": "cais/mmlu (all, test)",
        "n_questions": args.n,
        "sample_seed": args.seed,
        "bootstrap": {"resamples": 1000, "seed": 42},
        "models": args.models,
        "judges": args.judges,
        "provider_base_url": args.base_url,
        "metrics": metrics,
        "judge_agreement": {
            "cohen_kappa_between_judges": cohen_kappa(a, b),
            "items_labelled_by_both": len(valid),
            "judge_accuracy_vs_ground_truth": judge_accuracy,
        },
        "notes": "Judges are the same models being evaluated unless --judges is "
                 "changed; this can bias judgments.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nSaved {out / 'summary.json'}")


if __name__ == "__main__":
    main()
