# Resources Catalog

## Summary

This document catalogs all resources gathered for the research project "'Most Underrated' as a Novelty Metric". The project investigates whether LLMs struggle to generate truly novel outputs when asked to name the most underrated item in a category, and whether this serves as a diagnostic for LLM creativity limitations.

## Papers

Total papers downloaded: 12

| # | Title | Authors | Year | File | Key Info |
|---|-------|---------|------|------|----------|
| 1 | Generative Monoculture in LLMs | Wu, Black, Chandrasekaran | 2024 | `papers/2407.02209_generative_monoculture.pdf` | Defines monoculture; book reviews + code |
| 2 | Creative Homogeneity Across LLMs | Wenger, Kenett | 2025 | `papers/2501.19361_creative_homogeneity_across_llms.pdf` | Cross-model homogeneity using AUT/DAT/FF |
| 3 | CreativityPrism | Hou et al. | 2026 | `papers/2510.20091_creativity_prism.pdf` | Quality/novelty/diversity framework |
| 4 | Beyond Divergent Creativity | Various | 2026 | `papers/2601.20546_beyond_divergent_creativity.pdf` | CDAT; alignment vs. novelty tradeoff |
| 5 | Divergent Creativity in Humans and LLMs | Various | 2024 | `papers/2405.13012_divergent_creativity_humans_llms.pdf` | 100K human DAT baseline |
| 6 | LiveIdeaBench | Various | 2024 | `papers/2412.17596_liveideabench.pdf` | Minimal-context divergent thinking eval |
| 7 | Encouraging Divergent Thinking | Various | 2023 | `papers/2305.19118_encouraging_divergent_thinking.pdf` | Multi-agent debate for creativity |
| 8 | Human Creativity in the Age of LLMs | Various | 2024 | `papers/2410.03703_human_creativity_age_of_llms.pdf` | Randomized experiments on LLM + human creativity |
| 9 | Correlated Errors in LLMs | Various | 2025 | `papers/2506.07962_correlated_errors_llms.pdf` | Cross-model correlation; RLHF worsens monoculture |
| 10 | CulturalBench | Various | 2024 | `papers/2410.02677_culturalbench.pdf` | Cultural knowledge gaps in LLMs |
| 11 | Improving Diversity of Commonsense | Various | 2024 | `papers/2404.16807_improving_diversity_commonsense.pdf` | ICL methods for diverse generation |
| 12 | Divergent-Convergent Thinking | Various | 2025 | `papers/2512.23601_divergent_convergent_thinking.pdf` | CreativeDC prompting framework |

See `papers/README.md` for detailed descriptions.

## Datasets

Total datasets: 2

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| Most Underrated Diagnostic | Custom | 120 prompts, 40 categories | Novelty/diversity probing | `datasets/underrated_diagnostic/` | Core experiment dataset |
| LLM Global Opinions | Anthropic/HuggingFace | 2,556 questions | Opinion diversity reference | `datasets/llm_global_opinions/` | Reference for opinion diversity |

See `datasets/README.md` for detailed descriptions and download instructions.

## Code Repositories

Total repositories cloned: 2

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| lechmazur/divergent | https://github.com/lechmazur/divergent | LLM Divergent Thinking Benchmark | `code/lechmazur-divergent/` | Pre-computed scores for 20+ LLMs |
| jayolson/divergent-association-task | https://github.com/jayolson/divergent-association-task | DAT scoring implementation | `code/divergent-association-task/` | Reference implementation with GloVe |

See `code/README.md` for detailed descriptions.

## Key External Resources (Not Downloaded)

| Resource | URL | Why Relevant |
|----------|-----|--------------|
| Gwern's Creative Benchmark | https://gwern.net/creative-benchmark | Comprehensive analysis of LLM diversity/creativity benchmarking gaps |
| CreativityPrism Project | https://joeyhou.github.io/CreativityPrism/ | Evaluation framework code and data |
| Anthropic Global Values Viz | https://llmglobalvalues.anthropic.com/ | Interactive tool for LLM opinion analysis |

## Resource Gathering Notes

### Search Strategy
1. Used WebSearch across multiple queries: LLM novelty/creativity evaluation, generative monoculture, output homogeneity, divergent thinking, cultural knowledge, subjective preferences
2. Searched arXiv, Semantic Scholar, HuggingFace, GitHub
3. Identified key research clusters: monoculture/homogeneity, creativity benchmarks, divergent thinking, cultural/opinion bias

### Selection Criteria
- Papers directly studying LLM output diversity and novelty limitations
- Creativity evaluation frameworks with automatic metrics
- Datasets for probing subjective/opinion-based LLM behavior
- Code repositories with divergent thinking evaluation implementations

### Challenges Encountered
- Paper-finder service was unavailable; manual search via WebSearch was effective
- No existing dataset specifically tests "most underrated" reasoning — required creating a custom dataset
- Most creativity benchmarks focus on word-level divergent thinking (DAT) rather than domain-specific knowledge/taste

### Gaps and Workarounds
- **No human baseline for "most underrated" questions**: Could be gathered via survey or Reddit scraping in experiment phase
- **No pre-existing "underrated" annotation**: Must be inferred from answer frequency/popularity
- **Limited cultural diversity data for taste**: CulturalBench covers factual knowledge, not aesthetic preferences

## Recommendations for Experiment Design

1. **Primary dataset**: Custom "Most Underrated" diagnostic (120 prompts, 40 categories, 3 templates)
2. **Baseline methods**:
   - Cross-model agreement rate (how often LLMs converge on same answer)
   - Intra-model diversity (within-model response variation across runs)
   - Answer popularity scoring (how well-known/frequently-cited the named item is)
3. **Evaluation metrics**:
   - Agreement rate (% convergence across models and runs)
   - Novelty paradox score (underrated items that are most-frequently-cited = paradox)
   - Semantic diversity (embedding distance between responses)
   - Category difficulty correlation (do harder categories show more or less diversity?)
4. **Code to adapt/reuse**:
   - `jayolson/divergent-association-task` for semantic distance scoring
   - Embedding models (sentence-transformers) for response clustering
   - LLM API clients for multi-model querying
5. **Models to test**: GPT-4/4o, Claude 3.5/Opus, Gemini Pro, Llama 3, Mistral, Command R+ (cover major families)
