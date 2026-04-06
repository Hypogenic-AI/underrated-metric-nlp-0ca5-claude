# Downloaded Datasets

This directory contains datasets for the research project "'Most Underrated' as a Novelty Metric".
Data files are NOT committed to git due to size. Follow the download instructions below.

## Dataset 1: Most Underrated Diagnostic Dataset (Custom)

### Overview
- **Source**: Custom-designed for this research
- **Size**: 120 prompts across 40 categories and 3 prompt templates
- **Format**: JSON
- **Task**: Probing LLM novelty/diversity in subjective recommendations
- **Location**: `datasets/underrated_diagnostic/prompts.json`
- **License**: Research use

### Description
A set of prompts asking LLMs to name the "most underrated" item in various categories
(e.g., "most underrated rock band", "most underrated cuisine"). Designed to measure:
1. **Agreement rate**: How often different models/runs give the same answer
2. **Novelty**: How "obvious" vs. "surprising" the answer is
3. **Diversity**: Spread of answers across the category space

### Domains
- Music (5 categories)
- Film & TV (5 categories)
- Literature (5 categories)
- Food & Drink (5 categories)
- Science & Technology (5 categories)
- Sports & Games (5 categories)
- Geography & Travel (5 categories)
- General Knowledge (5 categories)

### Loading
```python
import json
with open("datasets/underrated_diagnostic/prompts.json") as f:
    dataset = json.load(f)
prompts = dataset["prompts"]  # List of 120 prompt dicts
```

### Notes
- This dataset is already committed to git (small JSON file)
- No download needed; it's generated as part of the research

---

## Dataset 2: Anthropic LLM Global Opinions

### Overview
- **Source**: [Anthropic/llm_global_opinions on HuggingFace](https://huggingface.co/datasets/Anthropic/llm_global_opinions)
- **Size**: 2,556 survey questions
- **Format**: HuggingFace Dataset
- **Task**: Evaluating LLM opinion representation and bias
- **Location**: `datasets/llm_global_opinions/`
- **License**: See HuggingFace dataset card

### Description
Survey questions from the Global Attitudes Survey (GAS) and World Values Survey (WVS),
adapted to probe LLM opinions on global issues. Useful as a reference for how LLMs
handle subjective/opinion questions and for comparing opinion diversity.

### Download Instructions

**Using HuggingFace (recommended):**
```python
from datasets import load_dataset
dataset = load_dataset("Anthropic/llm_global_opinions")
dataset.save_to_disk("datasets/llm_global_opinions")
```

### Loading the Dataset
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/llm_global_opinions")
# Access: dataset['train'][0]
# Columns: question, selections, options, source
```

### Sample Data
```json
{
  "question": "How much confidence do you have in the national government?",
  "selections": {"United States": "...", "Germany": "..."},
  "options": ["A great deal", "Some", "Not too much", "None at all"],
  "source": "GAS"
}
```

### Notes
- Contains subjective opinion questions — useful baseline for comparing LLM opinion diversity
- Paper: "Towards Measuring the Representation of Subjective Global Opinions in Language Models" (arXiv:2306.16388)
