"""
Analysis of "Most Underrated" experiment results.
Computes agreement rates, cross-model overlap, novelty paradox scores,
and semantic diversity metrics.
"""

import json
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

RESULTS_DIR = Path("/workspaces/underrated-metric-nlp-0ca5-claude/results")
RAW_DIR = RESULTS_DIR / "raw"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def normalize_answer(answer: str) -> str:
    """Normalize an answer for comparison."""
    if not answer:
        return ""
    # Take only first line (some models add explanation despite instructions)
    answer = answer.split("\n")[0].strip()
    # Remove quotes, asterisks, leading "A: " etc.
    answer = re.sub(r'^["\']|["\']$', '', answer)
    answer = re.sub(r'^\*+|\*+$', '', answer)
    answer = re.sub(r'^(A:|Answer:)\s*', '', answer, flags=re.IGNORECASE)
    # Lowercase, strip articles
    answer = answer.lower().strip()
    answer = re.sub(r'^(the|a|an)\s+', '', answer)
    return answer


def load_results():
    """Load experiment results from checkpoint file."""
    results = []
    checkpoint = RAW_DIR / "responses_checkpoint.jsonl"
    if checkpoint.exists():
        with open(checkpoint) as f:
            for line in f:
                r = json.loads(line)
                if r["status"] == "success" and r["answer"]:
                    r["normalized_answer"] = normalize_answer(r["answer"])
                    results.append(r)
    print(f"Loaded {len(results)} successful responses")
    return results


def compute_intra_model_agreement(results):
    """For each model+category, compute how often the most common answer appears."""
    df = pd.DataFrame(results)
    agreement_data = []

    for (model, category), group in df.groupby(["model", "category"]):
        answers = group["normalized_answer"].tolist()
        if len(answers) < 2:
            continue
        counter = Counter(answers)
        most_common_answer, most_common_count = counter.most_common(1)[0]
        agreement_rate = most_common_count / len(answers)
        unique_answers = len(counter)
        agreement_data.append({
            "model": model,
            "category": category,
            "domain": group["domain"].iloc[0],
            "difficulty": group["difficulty"].iloc[0],
            "agreement_rate": agreement_rate,
            "top_answer": most_common_answer,
            "top_answer_count": most_common_count,
            "unique_answers": unique_answers,
            "total_runs": len(answers),
        })

    return pd.DataFrame(agreement_data)


def compute_inter_model_overlap(results):
    """For each category, compute Jaccard overlap of top answers between model pairs."""
    df = pd.DataFrame(results)
    overlap_data = []

    for category, cat_group in df.groupby("category"):
        # Get top answer(s) per model for this category
        model_top_answers = {}
        for model, model_group in cat_group.groupby("model"):
            answers = model_group["normalized_answer"].tolist()
            counter = Counter(answers)
            # Take answers that appear at least twice (or top 3)
            top_answers = set([a for a, c in counter.most_common(3)])
            model_top_answers[model] = top_answers

        models = list(model_top_answers.keys())
        for m1, m2 in combinations(models, 2):
            s1, s2 = model_top_answers[m1], model_top_answers[m2]
            if len(s1 | s2) == 0:
                continue
            jaccard = len(s1 & s2) / len(s1 | s2)
            overlap_data.append({
                "category": category,
                "model1": m1,
                "model2": m2,
                "jaccard": jaccard,
                "shared_answers": list(s1 & s2),
                "domain": cat_group["domain"].iloc[0],
                "difficulty": cat_group["difficulty"].iloc[0],
            })

    return pd.DataFrame(overlap_data)


def compute_novelty_paradox(results):
    """
    For each category, find the most popular answer across ALL models.
    If the same "underrated" item is the top pick for most models, it's paradoxically popular.
    """
    df = pd.DataFrame(results)
    paradox_data = []

    for category, cat_group in df.groupby("category"):
        # Get top answer per model
        model_tops = {}
        for model, model_group in cat_group.groupby("model"):
            counter = Counter(model_group["normalized_answer"].tolist())
            model_tops[model] = counter.most_common(1)[0][0]

        # Count how many models agree on the same top answer
        top_answer_counts = Counter(model_tops.values())
        most_common_across, count = top_answer_counts.most_common(1)[0]
        num_models = len(model_tops)

        # Also count total frequency of the most common cross-model answer
        all_answers = cat_group["normalized_answer"].tolist()
        total_freq = sum(1 for a in all_answers if a == most_common_across)

        paradox_data.append({
            "category": category,
            "domain": cat_group["domain"].iloc[0],
            "difficulty": cat_group["difficulty"].iloc[0],
            "consensus_answer": most_common_across,
            "models_agreeing": count,
            "total_models": num_models,
            "paradox_score": count / num_models,  # 1.0 = all models agree
            "total_frequency": total_freq,
            "total_responses": len(all_answers),
            "frequency_rate": total_freq / len(all_answers),
        })

    return pd.DataFrame(paradox_data)


def compute_semantic_diversity(results):
    """Compute semantic diversity using sentence-transformers embeddings."""
    from sentence_transformers import SentenceTransformer

    print("Loading sentence-transformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    df = pd.DataFrame(results)
    diversity_data = []

    for (model_name, category), group in df.groupby(["model", "category"]):
        answers = group["normalized_answer"].unique().tolist()
        if len(answers) < 2:
            diversity_data.append({
                "model": model_name,
                "category": category,
                "domain": group["domain"].iloc[0],
                "difficulty": group["difficulty"].iloc[0],
                "mean_pairwise_distance": 0.0,
                "num_unique": len(answers),
            })
            continue

        embeddings = model.encode(answers)
        sim_matrix = cosine_similarity(embeddings)
        # Mean pairwise distance (1 - similarity) for off-diagonal elements
        n = len(answers)
        distances = []
        for i in range(n):
            for j in range(i + 1, n):
                distances.append(1 - sim_matrix[i][j])

        diversity_data.append({
            "model": model_name,
            "category": category,
            "domain": group["domain"].iloc[0],
            "difficulty": group["difficulty"].iloc[0],
            "mean_pairwise_distance": np.mean(distances),
            "num_unique": len(answers),
        })

    return pd.DataFrame(diversity_data)


def plot_intra_model_agreement(agreement_df):
    """Plot intra-model agreement rates."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Agreement rate by model
    model_means = agreement_df.groupby("model")["agreement_rate"].agg(["mean", "std"]).reset_index()
    model_means = model_means.sort_values("mean", ascending=False)
    ax = axes[0]
    bars = ax.bar(range(len(model_means)), model_means["mean"], yerr=model_means["std"],
                  capsize=5, color=sns.color_palette("Set2", len(model_means)), edgecolor="black")
    ax.set_xticks(range(len(model_means)))
    ax.set_xticklabels(model_means["model"], rotation=30, ha="right")
    ax.set_ylabel("Mean Agreement Rate")
    ax.set_title("Intra-Model Agreement Rate\n(Higher = Less Diverse)")
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="50% threshold")
    ax.legend()

    # Plot 2: Agreement rate by difficulty
    diff_order = ["easy", "medium", "hard"]
    diff_data = agreement_df[agreement_df["difficulty"].isin(diff_order)]
    sns.boxplot(data=diff_data, x="difficulty", y="agreement_rate", order=diff_order, ax=axes[1],
                palette="RdYlGn_r")
    axes[1].set_ylabel("Agreement Rate")
    axes[1].set_title("Agreement Rate by Category Difficulty")
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "intra_model_agreement.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: intra_model_agreement.png")


def plot_novelty_paradox(paradox_df):
    """Plot the novelty paradox scores."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Paradox score distribution
    paradox_sorted = paradox_df.sort_values("paradox_score", ascending=False).head(20)
    ax = axes[0]
    colors = ["#d32f2f" if s >= 0.8 else "#ff9800" if s >= 0.6 else "#4caf50"
              for s in paradox_sorted["paradox_score"]]
    ax.barh(range(len(paradox_sorted)), paradox_sorted["paradox_score"], color=colors, edgecolor="black")
    ax.set_yticks(range(len(paradox_sorted)))
    labels = [f"{row.category}\n({row.consensus_answer})"
              for _, row in paradox_sorted.iterrows()]
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Paradox Score (fraction of models agreeing)")
    ax.set_title("Novelty Paradox: Top 20 Categories\n(Red = All Models Give Same 'Underrated' Answer)")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()

    # Plot 2: Paradox by domain
    domain_means = paradox_df.groupby("domain")["paradox_score"].agg(["mean", "std"]).reset_index()
    domain_means = domain_means.sort_values("mean", ascending=False)
    axes[1].bar(range(len(domain_means)), domain_means["mean"], yerr=domain_means["std"],
                capsize=5, color=sns.color_palette("Set3", len(domain_means)), edgecolor="black")
    axes[1].set_xticks(range(len(domain_means)))
    axes[1].set_xticklabels(domain_means["domain"], rotation=30, ha="right")
    axes[1].set_ylabel("Mean Paradox Score")
    axes[1].set_title("Novelty Paradox by Domain")
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "novelty_paradox.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: novelty_paradox.png")


def plot_cross_model_heatmap(results):
    """Create a heatmap of inter-model agreement."""
    df = pd.DataFrame(results)
    models = sorted(df["model"].unique())

    # For each model pair, compute fraction of categories where top answers overlap
    overlap_matrix = np.zeros((len(models), len(models)))

    for category, cat_group in df.groupby("category"):
        model_tops = {}
        for model, model_group in cat_group.groupby("model"):
            counter = Counter(model_group["normalized_answer"].tolist())
            model_tops[model] = counter.most_common(1)[0][0]

        for i, m1 in enumerate(models):
            for j, m2 in enumerate(models):
                if m1 in model_tops and m2 in model_tops:
                    if model_tops[m1] == model_tops[m2]:
                        overlap_matrix[i][j] += 1

    num_categories = len(df["category"].unique())
    overlap_matrix /= num_categories

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(overlap_matrix, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=models, yticklabels=models, ax=ax,
                vmin=0, vmax=1, square=True)
    ax.set_title("Cross-Model Top Answer Agreement Rate\n(Fraction of Categories with Same #1 Answer)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "cross_model_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: cross_model_heatmap.png")


def plot_unique_answers_distribution(agreement_df):
    """Plot distribution of unique answers per category per model."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for model in sorted(agreement_df["model"].unique()):
        model_data = agreement_df[agreement_df["model"] == model]
        counts = model_data["unique_answers"].value_counts().sort_index()
        ax.plot(counts.index, counts.values, marker="o", label=model)
    ax.set_xlabel("Number of Unique Answers (out of 10 runs)")
    ax.set_ylabel("Number of Categories")
    ax.set_title("Distribution of Answer Diversity Across Categories")
    ax.legend()
    ax.set_xticks(range(1, 11))
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "unique_answers_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: unique_answers_distribution.png")


def plot_semantic_diversity(diversity_df):
    """Plot semantic diversity by model and domain."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.boxplot(data=diversity_df, x="model", y="mean_pairwise_distance", ax=axes[0],
                palette="Set2")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=30, ha="right")
    axes[0].set_ylabel("Mean Pairwise Semantic Distance")
    axes[0].set_title("Semantic Diversity of Answers by Model")

    sns.boxplot(data=diversity_df, x="domain", y="mean_pairwise_distance", ax=axes[1],
                palette="Set3")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=30, ha="right")
    axes[1].set_ylabel("Mean Pairwise Semantic Distance")
    axes[1].set_title("Semantic Diversity by Domain")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "semantic_diversity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: semantic_diversity.png")


def generate_summary_table(agreement_df, paradox_df, diversity_df):
    """Generate a summary statistics table."""
    summary = {}

    # Per-model stats
    for model in agreement_df["model"].unique():
        m_agree = agreement_df[agreement_df["model"] == model]
        m_div = diversity_df[diversity_df["model"] == model] if diversity_df is not None else None
        summary[model] = {
            "mean_agreement_rate": m_agree["agreement_rate"].mean(),
            "std_agreement_rate": m_agree["agreement_rate"].std(),
            "mean_unique_answers": m_agree["unique_answers"].mean(),
            "categories_with_single_answer": (m_agree["unique_answers"] == 1).sum(),
        }
        if m_div is not None and len(m_div) > 0:
            summary[model]["mean_semantic_diversity"] = m_div["mean_pairwise_distance"].mean()

    summary_df = pd.DataFrame(summary).T
    summary_df.to_csv(RESULTS_DIR / "model_summary.csv")
    print("\nModel Summary:")
    print(summary_df.to_string())

    # Overall paradox stats
    print(f"\nNovelty Paradox Summary:")
    print(f"  Mean paradox score: {paradox_df['paradox_score'].mean():.3f}")
    print(f"  Categories where ALL models agree: {(paradox_df['paradox_score'] == 1.0).sum()}/{len(paradox_df)}")
    print(f"  Categories where >60% models agree: {(paradox_df['paradox_score'] >= 0.6).sum()}/{len(paradox_df)}")

    return summary_df


def run_statistical_tests(agreement_df, paradox_df):
    """Run statistical tests on the results."""
    from scipy import stats

    print("\n=== Statistical Tests ===\n")

    # Test 1: Is agreement rate significantly above chance?
    # For open-ended questions with ~100+ possible answers, chance agreement is ~0.01
    chance_rate = 0.1  # conservative estimate
    agreement_rates = agreement_df["agreement_rate"].values
    t_stat, p_value = stats.ttest_1samp(agreement_rates, chance_rate)
    print(f"H1: Agreement rate > chance ({chance_rate})")
    print(f"  Mean agreement rate: {agreement_rates.mean():.3f} ± {agreement_rates.std():.3f}")
    print(f"  t-statistic: {t_stat:.3f}, p-value: {p_value:.6f}")
    print(f"  Effect size (Cohen's d): {(agreement_rates.mean() - chance_rate) / agreement_rates.std():.3f}")
    print()

    # Test 2: Difficulty effect on agreement
    diff_groups = {d: agreement_df[agreement_df["difficulty"] == d]["agreement_rate"].values
                   for d in ["easy", "medium", "hard"]
                   if len(agreement_df[agreement_df["difficulty"] == d]) > 0}
    if len(diff_groups) >= 2:
        stat, p = stats.kruskal(*diff_groups.values())
        print(f"H4: Difficulty affects agreement rate")
        for d, vals in diff_groups.items():
            print(f"  {d}: {vals.mean():.3f} ± {vals.std():.3f}")
        print(f"  Kruskal-Wallis H: {stat:.3f}, p-value: {p:.6f}")
    print()

    # Test 3: Paradox score > 0.5 (majority agreement)
    paradox_scores = paradox_df["paradox_score"].values
    t_stat2, p_value2 = stats.ttest_1samp(paradox_scores, 0.4)
    print(f"Novelty Paradox: Cross-model consensus > 0.4")
    print(f"  Mean paradox score: {paradox_scores.mean():.3f} ± {paradox_scores.std():.3f}")
    print(f"  t-statistic: {t_stat2:.3f}, p-value: {p_value2:.6f}")

    return {
        "agreement_vs_chance": {"t": t_stat, "p": p_value, "mean": agreement_rates.mean()},
        "paradox_vs_threshold": {"t": t_stat2, "p": p_value2, "mean": paradox_scores.mean()},
    }


def main():
    results = load_results()
    if not results:
        print("No results found. Run the experiment first.")
        return

    print(f"\n{'='*60}")
    print("ANALYSIS: 'Most Underrated' as a Novelty Metric")
    print(f"{'='*60}\n")

    # 1. Intra-model agreement
    print("--- Computing intra-model agreement ---")
    agreement_df = compute_intra_model_agreement(results)
    plot_intra_model_agreement(agreement_df)
    plot_unique_answers_distribution(agreement_df)

    # 2. Novelty paradox
    print("\n--- Computing novelty paradox ---")
    paradox_df = compute_novelty_paradox(results)
    plot_novelty_paradox(paradox_df)

    # 3. Cross-model heatmap
    print("\n--- Computing cross-model overlap ---")
    plot_cross_model_heatmap(results)

    # 4. Inter-model overlap
    overlap_df = compute_inter_model_overlap(results)
    if len(overlap_df) > 0:
        print(f"Mean Jaccard overlap: {overlap_df['jaccard'].mean():.3f}")

    # 5. Semantic diversity
    print("\n--- Computing semantic diversity ---")
    diversity_df = compute_semantic_diversity(results)
    plot_semantic_diversity(diversity_df)

    # 6. Summary and statistics
    print("\n--- Generating summary ---")
    summary_df = generate_summary_table(agreement_df, paradox_df, diversity_df)
    stats_results = run_statistical_tests(agreement_df, paradox_df)

    # Save all analysis outputs
    agreement_df.to_csv(RESULTS_DIR / "agreement_analysis.csv", index=False)
    paradox_df.to_csv(RESULTS_DIR / "paradox_analysis.csv", index=False)
    if len(overlap_df) > 0:
        overlap_df.to_csv(RESULTS_DIR / "overlap_analysis.csv", index=False)
    diversity_df.to_csv(RESULTS_DIR / "diversity_analysis.csv", index=False)

    with open(RESULTS_DIR / "stats_tests.json", "w") as f:
        json.dump(stats_results, f, indent=2, default=str)

    # Print top paradox categories
    print("\n--- Top Novelty Paradox Categories ---")
    top_paradox = paradox_df.sort_values("paradox_score", ascending=False).head(15)
    for _, row in top_paradox.iterrows():
        print(f"  {row.category}: '{row.consensus_answer}' "
              f"(score={row.paradox_score:.2f}, {row.models_agreeing}/{row.total_models} models)")

    # Print example answers per category (first 5 categories)
    print("\n--- Example Answers by Category ---")
    df = pd.DataFrame(results)
    for category in sorted(df["category"].unique())[:10]:
        cat_data = df[df["category"] == category]
        print(f"\n  Category: {category}")
        for model in sorted(cat_data["model"].unique()):
            model_data = cat_data[cat_data["model"] == model]
            counter = Counter(model_data["normalized_answer"].tolist())
            top3 = counter.most_common(3)
            top3_str = ", ".join([f"'{a}' ({c}x)" for a, c in top3])
            print(f"    {model}: {top3_str}")

    print(f"\nAll analysis outputs saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
