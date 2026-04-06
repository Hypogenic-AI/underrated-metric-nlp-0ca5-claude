# Research Plan: "Most Underrated" as a Novelty Metric

## Motivation & Novelty Assessment

### Why This Research Matters
LLMs are increasingly used for creative tasks — brainstorming, ideation, recommendation. If they systematically produce "obvious" answers to subjective questions, this limits their utility as creative partners. The "most underrated X" question is a sharp diagnostic because a truly novel answer should, by definition, NOT be the consensus answer — yet if all LLMs converge on the same "underrated" pick, that pick is paradoxically the most popular "underrated" choice, not genuinely underrated at all.

### Gap in Existing Work
Literature on generative monoculture (Wu et al. 2024), creative homogeneity (Wenger & Kenett 2025), and correlated errors (2025) establishes that LLMs produce narrower output distributions than humans. However:
- No study specifically tests **subjective judgment about underratedness** — a task requiring reasoning about both quality AND popularity
- Existing creativity benchmarks (DAT, AUT) measure divergent thinking with abstract stimuli, not domain-specific taste/knowledge
- The "novelty paradox" (an "underrated" answer that everyone gives isn't truly underrated) has not been formally measured

### Our Novel Contribution
1. **A new diagnostic task**: "Most underrated X" as a probe for LLM novelty capacity
2. **The Novelty Paradox Score**: Quantifying when "underrated" answers are paradoxically popular across models
3. **Cross-model convergence analysis**: Measuring whether different LLM families give the same "underrated" picks
4. **Reddit baseline comparison**: Comparing LLM answers against crowd-sourced "obvious underrated" answers

### Experiment Justification
- **Experiment 1 (Multi-model querying)**: Needed to measure inter-model convergence — do all models give the same answer?
- **Experiment 2 (Within-model diversity)**: Needed to measure intra-model diversity — does a single model give varied answers across runs?
- **Experiment 3 (Reddit baseline comparison)**: Needed to test whether LLM answers match the "obvious" underrated picks from human crowds
- **Experiment 4 (Semantic clustering)**: Needed to quantify answer diversity beyond exact-match agreement

## Research Question
Do LLMs converge on "obviously underrated" answers when asked "What is the most underrated X?", and can this convergence serve as a diagnostic metric for LLM novelty limitations?

## Hypothesis Decomposition
H1: Different LLMs will converge on the same "most underrated" answers at rates significantly above chance
H2: Within a single model, repeated queries will produce low answer diversity (high intra-model agreement)
H3: LLM answers will significantly overlap with commonly cited "underrated" items from Reddit/web sources
H4: Cross-model convergence will be higher for "easy" categories than "hard" ones

## Proposed Methodology

### Approach
Query 5 LLMs across 40 categories, 10 runs each, using the simplest template. Compare answers within models, across models, and against web-sourced "obvious underrated" baselines.

### Models
Via OpenAI API: GPT-4.1
Via OpenRouter: Claude Sonnet 4.5, Gemini 2.5 Pro, Llama 4 Maverick, Mistral Large

### Experimental Steps
1. Query each model 10x per category (40 categories × 5 models × 10 runs = 2000 calls)
2. Extract the specific named item from each response
3. Normalize answers (lowercase, remove articles, fuzzy matching)
4. Compute intra-model agreement rate per category
5. Compute inter-model Jaccard overlap
6. Scrape/search for Reddit "obvious underrated" answers per category
7. Compare LLM answers against Reddit baselines
8. Compute semantic similarity using sentence-transformers embeddings
9. Analyze by domain and difficulty

### Baselines
- Random: uniform distribution (expected agreement = 0 for open-ended questions)
- Reddit consensus: most frequently cited "underrated" items from web searches

### Evaluation Metrics
1. **Intra-model agreement rate**: % of runs giving the same top answer (higher = less diverse)
2. **Inter-model Jaccard overlap**: |intersection|/|union| of answer sets across model pairs
3. **Novelty Paradox Score**: For each category, % of models whose top answer is the same entity
4. **Reddit overlap rate**: % of LLM top answers that match Reddit's most common "underrated" picks
5. **Semantic diversity**: Mean pairwise cosine distance of answer embeddings within/across models

### Statistical Analysis Plan
- Chi-squared test for agreement rates vs. chance
- Bootstrap confidence intervals for overlap metrics
- Kruskal-Wallis test for difficulty-level differences
- Significance level: α = 0.05

## Expected Outcomes
- H1 supported: Inter-model Jaccard > 0.3 (meaningful overlap in open-ended answers)
- H2 supported: Intra-model agreement > 50% for most categories
- H3 supported: >60% of LLM top answers appear in Reddit "obvious" lists
- H4 supported: Easy categories show higher convergence than hard ones

## Timeline
- Phase 2 (Setup): 15 min
- Phase 3 (Implementation): 45 min
- Phase 4 (Experiments): 60 min (API calls + Reddit search)
- Phase 5 (Analysis): 30 min
- Phase 6 (Documentation): 20 min

## Potential Challenges
- API rate limits → use exponential backoff, stagger requests
- Answer extraction from verbose responses → use structured prompting ("Answer in format: [ANSWER]")
- Fuzzy matching of equivalent answers → use embedding similarity threshold
- Reddit data quality → manually verify a sample

## Success Criteria
The research succeeds if we can quantitatively demonstrate that LLMs converge on "obvious underrated" answers, establishing the "most underrated X" question as a valid novelty diagnostic.
