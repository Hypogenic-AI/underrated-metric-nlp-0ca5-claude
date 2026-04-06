# "Most Underrated" as a Novelty Metric for LLMs

Testing whether LLMs can generate genuinely novel answers to subjective questions, or whether they converge on "obviously underrated" consensus picks.

## Key Findings

- **LLMs repeat themselves**: Mean 68.2% intra-model agreement rate across 40 categories (Cohen's d = 2.42 vs. chance, p < 0.0001). Claude Sonnet gave identical answers in all 10 runs for 13/40 categories.
- **Models agree with each other**: GPT-4.1 and Claude share the same top "underrated" pick in 23% of categories — far above chance for open-ended questions.
- **Answers match Reddit consensus**: 49.4% of LLM top answers match commonly cited "underrated" picks from online forums. GPT-4.1 matches Reddit 75% of the time.
- **The Novelty Paradox**: When all LLMs name the same "underrated" item (e.g., "Philosophy of Technology" as most underrated philosophy branch), that item is paradoxically the most popular "underrated" pick — not genuinely underrated at all.
- **Model family matters**: Llama 4 Maverick (20% Reddit match) and Qwen3-235B (5.3 unique answers/category) show more independence than GPT-4.1 and Claude.

## How to Reproduce

```bash
uv venv && source .venv/bin/activate
uv add openai httpx numpy pandas matplotlib seaborn scikit-learn sentence-transformers tqdm
export OPENAI_API_KEY=... OPENROUTER_KEY=...

python src/run_experiment.py        # Query 5 LLMs x 40 categories x 10 runs
python src/collect_reddit_baseline.py  # Collect "obvious underrated" baselines
python src/analyze_results.py       # Compute metrics and generate plots
python src/reddit_comparison.py     # Compare against Reddit baselines
```

## File Structure

```
├── REPORT.md                    # Full research report with results
├── planning.md                  # Research plan and methodology
├── src/
│   ├── run_experiment.py        # Main experiment: multi-model querying
│   ├── analyze_results.py       # Analysis: agreement, paradox, diversity
│   ├── collect_reddit_baseline.py  # Reddit "obvious" baseline collection
│   └── reddit_comparison.py     # LLM vs Reddit comparison
├── datasets/
│   └── underrated_diagnostic/   # 40-category diagnostic prompt dataset
├── results/
│   ├── raw/                     # Raw model responses (1,716 total)
│   ├── plots/                   # Visualizations
│   ├── agreement_analysis.csv   # Per-model per-category agreement rates
│   ├── paradox_analysis.csv     # Novelty paradox scores
│   └── reddit_comparison.csv    # Reddit overlap analysis
├── papers/                      # Related research papers
├── literature_review.md         # Literature review
└── resources.md                 # Resource catalog
```

## Models Tested

| Model | Provider | Responses | Agreement Rate | Reddit Match |
|-------|----------|-----------|---------------|-------------|
| GPT-4.1 | OpenAI | 400 | 64.8% | 75.0% |
| Claude Sonnet 4.5 | Anthropic | 400 | 72.5% | 60.0% |
| Gemini 2.0 Flash | Google | 400 | 74.0% | 42.5% |
| Llama 4 Maverick | Meta | 400 | 67.0% | 20.0% |
| Qwen3-235B | Alibaba | 116 | 50.3% | 50.0% |

See [REPORT.md](REPORT.md) for full analysis and discussion.
