# Cloned Repositories

## Repo 1: lechmazur/divergent
- **URL**: https://github.com/lechmazur/divergent
- **Purpose**: LLM Divergent Thinking Creativity Benchmark. Tests LLMs' ability to generate 25 unique words unrelated to 50 initial random words. Contains evaluation data across many LLMs.
- **Location**: `code/lechmazur-divergent/`
- **Key files**: 
  - `initial_list_sorted.txt` — the 50 seed words
  - Various model output files with LLM responses
  - Evaluation scores for multiple models
- **Notes**: Extensive dataset of LLM divergent thinking responses; useful as baseline for measuring novelty. Contains pre-computed scores for 20+ LLMs.

## Repo 2: jayolson/divergent-association-task
- **URL**: https://github.com/jayolson/divergent-association-task
- **Purpose**: Reference implementation of the Divergent Association Task (DAT) — a brief measure of creativity that computes semantic distance between words.
- **Location**: `code/divergent-association-task/`
- **Key files**:
  - `examples.py` — example usage
  - Core scoring implementation using GloVe embeddings
- **Notes**: Can be used to score word-level diversity in LLM outputs. The DAT score is a standard metric in creativity research.
