# Literature Review: "Most Underrated" as a Novelty Metric

## Research Area Overview

This review examines the intersection of LLM output diversity, creativity evaluation, and the phenomenon of "generative monoculture" — the tendency of large language models to produce narrower, more homogeneous outputs than the diversity present in their training data. The core research question is whether asking LLMs to name the "most underrated" item in a category X can serve as a diagnostic for genuine novelty and reasoning depth, or whether LLMs default to conventionally "safe" answers that many humans would also consider underrated (i.e., the obvious underrated choices).

## Key Papers

### Paper 1: Generative Monoculture in Large Language Models
- **Authors**: Fan Wu, Emily Black, Varun Chandrasekaran (UIUC, Barnard)
- **Year**: 2024
- **Source**: arXiv:2407.02209
- **Key Contribution**: Defines "generative monoculture" as a distribution shift from source data (human-generated) to model output, where the output distribution is statistically narrower. Formalizes as: Dispersion(h(x)|x ~ P_gen) < Dispersion(h(x)|x ~ P_src).
- **Methodology**: Compares diversity of LLM-generated book reviews (sentiment) and code solutions (algorithm variety) against human-generated baselines (Goodreads dataset, competitive programming). Uses dispersion metrics including entropy and range.
- **Datasets Used**: Goodreads book reviews, competitive programming solutions (Codeforces/LeetCode)
- **Results**: LLM-generated reviews overwhelmingly positive (sentiment > 0.85) vs. human reviews spanning 0-1. Code solutions use narrower range of algorithms. Changing temperature, sampling strategies, and prompts are insufficient to mitigate monoculture. Root cause likely in RLHF alignment.
- **Code Available**: No
- **Relevance to Our Research**: Directly supports hypothesis that LLMs narrow diversity. The "most underrated" prompt is essentially testing whether LLMs can escape monoculture in subjective domains — if they can't, their answers will cluster around "obviously underrated" items.

### Paper 2: We're Different, We're the Same: Creative Homogeneity Across LLMs
- **Authors**: Emily Wenger (Duke), Yoed Kenett (Technion)
- **Year**: 2025
- **Source**: arXiv:2501.19361
- **Key Contribution**: Demonstrates that creative homogeneity extends *across* different LLMs, not just within a single model. LLM responses to creativity tests are far more similar to other LLM responses than human responses are to each other.
- **Methodology**: Uses standardized creativity tests — Alternative Uses Task (AUT), Forward Flow (FF), and Divergent Association Task (DAT) — to elicit responses from multiple LLMs and humans. Measures population-level semantic similarity.
- **Datasets Used**: Human responses from existing AUT/FF/DAT studies
- **Results**: LLMs match/outperform humans on individual creativity scores but show much higher inter-response similarity. Even altering system prompts for higher creativity only slightly increases variability, still below human levels. Feature space universality across models contributes to cross-model homogeneity.
- **Code Available**: No
- **Relevance to Our Research**: If all LLMs produce similar "most underrated" answers, this confirms the cross-model homogeneity finding and validates our metric as a novelty diagnostic.

### Paper 3: CreativityPrism: A Holistic Evaluation Framework for LLM Creativity
- **Authors**: Zhaoyi Joey Hou et al. (Pittsburgh, JHU, Notre Dame, AI2, UNC, UMass)
- **Year**: 2026
- **Source**: arXiv:2510.20091
- **Key Contribution**: Comprehensive framework decomposing creativity into quality, novelty, and diversity across 8 tasks in 3 domains (divergent thinking, creative writing, logical reasoning) with 17 metrics.
- **Methodology**: Evaluates 17 SoTA LLMs with validated automatic evaluation judges. Uses AUT, DAT, TTCT, short story writing, creative math, NeoCoder tasks.
- **Results**: Proprietary LLMs lead by 15% in creative writing and logical reasoning, but NO significant advantage in divergent thinking. Novelty metrics show weak or negative correlations with other metrics — high novelty doesn't transfer across domains.
- **Code Available**: Yes — https://joeyhou.github.io/CreativityPrism/
- **Relevance to Our Research**: The finding that novelty is orthogonal to other creativity dimensions validates the "most underrated" test as measuring something distinct. The divergent thinking gap (no proprietary advantage) suggests this is an under-explored area where our diagnostic could contribute.

### Paper 4: Correlated Errors in Large Language Models
- **Authors**: Various
- **Year**: 2025
- **Source**: arXiv:2506.07962
- **Key Contribution**: Shows that different LLMs are more correlated with each other than with ground truth, and monoculture is worse within model families and with RLHF training.
- **Results**: Changing inference parameters (temperature, top-p, prompts) does NOT mitigate monoculture. Severe intra-model monoculture on task instances.
- **Relevance to Our Research**: Explains why even "creative" prompting won't help LLMs give truly novel "most underrated" answers — the monoculture is structural, not a sampling artifact.

### Paper 5: Beyond Divergent Creativity
- **Authors**: Various
- **Year**: 2026
- **Source**: arXiv:2601.20546
- **Key Contribution**: Introduces Conditional Divergent Association Task (CDAT), measuring novelty conditional on contextual appropriateness. Finds smaller model families sometimes show more creativity, while advanced families favor appropriateness at lower novelty.
- **Relevance to Our Research**: Alignment/training shifts models toward "appropriate" (i.e., safe, popular) answers — directly explaining why LLMs would pick "obviously underrated" items rather than genuinely novel ones.

### Paper 6: Divergent Creativity in Humans and Large Language Models
- **Authors**: Various
- **Year**: 2024
- **Source**: arXiv:2405.13012
- **Key Contribution**: Benchmarks LLMs against 100K human responses on DAT and creative writing tasks. LLMs surpass average human performance but fall short of highly creative humans.
- **Relevance to Our Research**: Establishes that LLMs can be "averagely creative" but lack the tail of genuine novelty — exactly what we expect in "most underrated" responses.

### Paper 7: Evaluating LLMs' Divergent Thinking (LiveIdeaBench)
- **Authors**: Various
- **Year**: 2024/2025
- **Source**: arXiv:2412.17596
- **Key Contribution**: LiveIdeaBench uses single-keyword prompts to assess divergent thinking across originality, feasibility, fluency, flexibility, and clarity.
- **Relevance to Our Research**: Minimal-context prompting is analogous to our "most underrated X" format — tests what the model generates without heavy guidance.

## Common Methodologies

### Measuring Novelty/Diversity
- **Semantic similarity**: Cosine distance between embedding representations of outputs (used across multiple papers)
- **Divergent Association Task (DAT)**: Average semantic distance between generated words; higher = more creative
- **Dispersion metrics**: Entropy, variance, range of attribute distributions (generative monoculture paper)
- **Jaccard similarity**: For structured outputs like code algorithms
- **N-gram diversity**: Distinct n-grams as proportion of total
- **LLM-as-Judge**: For subjective metrics like originality and surprise

### Addressing Monoculture
- Temperature/sampling changes: **Insufficient** (multiple papers confirm)
- Multi-agent debate: **Partially effective** (Liang et al. 2023)
- System prompt engineering: **Marginally effective** (Wenger & Kenett 2025)
- Fundamental fix requires: Changes to alignment/fine-tuning paradigms

## Standard Baselines
- **Human responses**: Gold standard for diversity comparison
- **Random baseline**: Uniform sampling from category
- **Cross-model comparison**: Same prompt across 10+ LLMs
- **Multiple runs**: Same model, same prompt, N=50+ generations

## Evaluation Metrics for Our Research
1. **Agreement Rate**: Fraction of model responses that give the same answer (higher = more monoculture)
2. **Popularity Score**: How "famous" or "commonly cited" the named item is (e.g., Google search frequency)
3. **Semantic Clustering**: Embedding-based clustering of responses to measure diversity
4. **Surprisal/Novelty**: Inverse frequency of the answer in a reference corpus or across model responses
5. **Cross-Model Overlap**: Jaccard similarity of answer sets between different LLMs

## Datasets in the Literature
- **Goodreads book reviews**: Sentiment diversity baseline (Wu et al. 2024)
- **Codeforces/LeetCode solutions**: Algorithm diversity baseline
- **DAT human responses**: 100K+ humans (Olson et al. 2021)
- **AUT responses**: Standard creativity assessment data
- **CulturalBench**: 1,696 questions on cultural knowledge across 45 regions
- **GlobalOpinionQA (Anthropic)**: 2,556 survey questions from global attitude surveys

## Gaps and Opportunities

1. **No existing benchmark directly tests "underrated" reasoning**: While creativity benchmarks test divergent thinking and novelty, none specifically ask models to identify items that are undervalued relative to their quality. This requires a different kind of reasoning — understanding both popular opinion AND quality independently.

2. **Subjective domain monoculture is under-studied**: Most monoculture research focuses on objective tasks (code, reviews with clear sentiment). The "most underrated" question probes a purely subjective domain where there's no single correct answer, making monoculture especially problematic.

3. **Cross-model consensus as a monoculture signal**: If all major LLMs give the same "most underrated" answer, this is itself evidence that the answer isn't truly underrated — it's the consensus view. This creates a measurable paradox that can serve as a diagnostic.

4. **Connection to "taste" and cultural knowledge**: Gwern's analysis highlights that benchmarks punishing mode collapse are essentially absent. Our "most underrated" metric directly addresses this gap by testing whether models can demonstrate genuine taste vs. defaulting to popular consensus.

## Recommendations for Our Experiment

### Recommended Approach
1. **Prompt multiple LLMs** (5-10 models across families) with "What is the most underrated X?" for 40 categories
2. **Collect N=20+ responses per model** per category to measure within-model diversity
3. **Compute metrics**: agreement rate, semantic clustering, popularity scoring, cross-model overlap
4. **Compare against human baselines**: Survey or Reddit data for the same questions

### Recommended Metrics
- **Intra-model agreement**: How often the same model gives the same answer (lower = more diverse)
- **Inter-model agreement**: How often different models converge (measures cross-model monoculture)
- **Answer popularity**: Web frequency / familiarity of the named item
- **Novelty paradox score**: If an "underrated" answer is given by most models, it's paradoxically popular

### Methodological Considerations
- Use consistent temperature (default) across models for fair comparison
- Include both open-source (Llama, Mistral) and proprietary (GPT-4, Claude) models
- Control for prompt template variation (multiple phrasings)
- Consider category difficulty (easy categories like "vegetable" vs. hard like "chess opening")
