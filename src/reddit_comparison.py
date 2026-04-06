"""
Compare LLM answers against Reddit/web "obvious underrated" baselines.
"""

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = Path("/workspaces/underrated-metric-nlp-0ca5-claude/results")
RAW_DIR = RESULTS_DIR / "raw"
PLOTS_DIR = RESULTS_DIR / "plots"


def normalize(s):
    s = s.lower().strip()
    s = re.sub(r'^(the|a|an)\s+', '', s)
    s = re.sub(r'["\'\*]', '', s)
    return s


def fuzzy_match(answer, candidates):
    """Check if answer fuzzy-matches any candidate."""
    norm_answer = normalize(answer)
    for cand in candidates:
        norm_cand = normalize(cand)
        # Exact match
        if norm_answer == norm_cand:
            return True
        # Substring match (either direction)
        if norm_answer in norm_cand or norm_cand in norm_answer:
            return True
        # First word match for multi-word answers
        if norm_answer.split()[0] == norm_cand.split()[0] and len(norm_answer.split()[0]) > 3:
            return True
    return False


def main():
    # Load results
    results = []
    with open(RAW_DIR / "responses_checkpoint.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["status"] == "success" and r["answer"]:
                r["normalized_answer"] = normalize(r["answer"].split("\n")[0])
                results.append(r)

    # Load Reddit baselines
    with open(RAW_DIR / "reddit_baselines.json") as f:
        baselines = json.load(f)
    baseline_map = {b["category"]: b["obvious_underrated"] for b in baselines}

    df = pd.DataFrame(results)
    models = sorted(df["model"].unique())

    # For each model+category, check if top answer matches Reddit obvious picks
    comparison = []
    for (model, category), group in df.groupby(["model", "category"]):
        counter = Counter(group["normalized_answer"].tolist())
        top_answer = counter.most_common(1)[0][0]
        reddit_picks = baseline_map.get(category, [])

        matches_reddit = fuzzy_match(top_answer, reddit_picks) if reddit_picks else False

        comparison.append({
            "model": model,
            "category": category,
            "domain": group["domain"].iloc[0],
            "difficulty": group["difficulty"].iloc[0],
            "top_answer": top_answer,
            "reddit_picks": reddit_picks,
            "matches_reddit": matches_reddit,
        })

    comp_df = pd.DataFrame(comparison)

    # Overall match rate by model
    print("=== Reddit Baseline Comparison ===\n")
    print("Match rate (LLM top answer appears in Reddit's 'obvious underrated' list):\n")
    for model in models:
        m_data = comp_df[comp_df["model"] == model]
        rate = m_data["matches_reddit"].mean()
        count = m_data["matches_reddit"].sum()
        total = len(m_data)
        print(f"  {model}: {rate:.1%} ({int(count)}/{total} categories)")

    overall_rate = comp_df["matches_reddit"].mean()
    print(f"\n  Overall: {overall_rate:.1%}")

    # By domain
    print("\nMatch rate by domain:")
    for domain in sorted(comp_df["domain"].unique()):
        d_data = comp_df[comp_df["domain"] == domain]
        print(f"  {domain}: {d_data['matches_reddit'].mean():.1%}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Reddit overlap by model
    model_rates = comp_df.groupby("model")["matches_reddit"].mean().sort_values(ascending=False)
    colors = ["#d32f2f" if r > 0.5 else "#ff9800" if r > 0.3 else "#4caf50" for r in model_rates]
    axes[0].bar(range(len(model_rates)), model_rates.values, color=colors, edgecolor="black")
    axes[0].set_xticks(range(len(model_rates)))
    axes[0].set_xticklabels(model_rates.index, rotation=30, ha="right")
    axes[0].set_ylabel("Fraction Matching Reddit 'Obvious' Picks")
    axes[0].set_title("LLM Top Answers Matching Reddit\n(Higher = More 'Obvious')")
    axes[0].set_ylim(0, 1)
    axes[0].axhline(y=0.5, color="red", linestyle="--", alpha=0.5)

    # Plot 2: By domain
    domain_rates = comp_df.groupby("domain")["matches_reddit"].mean().sort_values(ascending=False)
    axes[1].bar(range(len(domain_rates)), domain_rates.values,
                color=sns.color_palette("Set3", len(domain_rates)), edgecolor="black")
    axes[1].set_xticks(range(len(domain_rates)))
    axes[1].set_xticklabels(domain_rates.index, rotation=30, ha="right")
    axes[1].set_ylabel("Fraction Matching Reddit")
    axes[1].set_title("Reddit Overlap by Domain")
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "reddit_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved: reddit_comparison.png")

    # Save detailed comparison
    comp_df.to_csv(RESULTS_DIR / "reddit_comparison.csv", index=False)

    # Print interesting examples
    print("\n=== Examples Where ALL Models Match Reddit ===")
    for category in sorted(comp_df["category"].unique()):
        cat_data = comp_df[comp_df["category"] == category]
        if cat_data["matches_reddit"].all():
            answers = cat_data[["model", "top_answer"]].values
            reddit = baseline_map.get(category, [])[:3]
            print(f"\n  {category}:")
            for model, answer in answers:
                print(f"    {model}: {answer}")
            print(f"    Reddit obvious: {reddit}")

    print("\n=== Examples Where NO Model Matches Reddit ===")
    for category in sorted(comp_df["category"].unique()):
        cat_data = comp_df[comp_df["category"] == category]
        if not cat_data["matches_reddit"].any():
            answers = cat_data[["model", "top_answer"]].values
            reddit = baseline_map.get(category, [])[:3]
            print(f"\n  {category}:")
            for model, answer in answers:
                print(f"    {model}: {answer}")
            print(f"    Reddit obvious: {reddit}")


if __name__ == "__main__":
    main()
