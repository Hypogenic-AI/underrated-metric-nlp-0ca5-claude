"""
Main experiment: Query multiple LLMs with "most underrated X" prompts.
Collects responses across models and runs for novelty analysis.
"""

import json
import os
import time
import random
import hashlib
from pathlib import Path
from datetime import datetime

import openai
import httpx

# Configuration
SEED = 42
random.seed(SEED)

RESULTS_DIR = Path("/workspaces/underrated-metric-nlp-0ca5-claude/results/raw")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NUM_RUNS = 10  # runs per model per category

# System prompt to get structured answers
SYSTEM_PROMPT = """You are answering a survey question. Give ONLY the name of your answer — no explanation, no justification, no preamble. Just the specific name/title.

Example:
Q: What is the most underrated vegetable?
A: Rutabaga

Q: What is the most underrated sci-fi movie?
A: Dark City"""

# Models configuration
MODELS = {
    "gpt-4.1": {
        "provider": "openai",
        "model_id": "gpt-4.1",
    },
    "claude-sonnet-4-5": {
        "provider": "openrouter",
        "model_id": "anthropic/claude-sonnet-4.5",
    },
    "gemini-2.0-flash": {
        "provider": "openrouter",
        "model_id": "google/gemini-2.0-flash-001",
    },
    "llama-4-maverick": {
        "provider": "openrouter",
        "model_id": "meta-llama/llama-4-maverick",
    },
    "qwen3-235b": {
        "provider": "openrouter",
        "model_id": "qwen/qwen3-235b-a22b",
    },
}


def get_openai_client():
    return openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def get_openrouter_client():
    return openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_KEY"],
    )


def query_model(model_name: str, prompt: str, run_id: int, max_retries: int = 3) -> dict:
    """Query a model and return the response with metadata."""
    config = MODELS[model_name]

    if config["provider"] == "openai":
        client = get_openai_client()
    else:
        client = get_openrouter_client()

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=config["model_id"],
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=1.0,
                max_tokens=100,
                timeout=30,
            )
            answer = response.choices[0].message.content
            if answer:
                answer = answer.strip()
            else:
                answer = ""
            return {
                "model": model_name,
                "prompt": prompt,
                "answer": answer,
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
                "status": "success",
            }
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return {
                "model": model_name,
                "prompt": prompt,
                "answer": None,
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
                "status": f"error: {str(e)}",
            }


def load_prompts():
    """Load prompts dataset, using only template_id=0 for consistency."""
    with open("datasets/underrated_diagnostic/prompts.json") as f:
        data = json.load(f)
    # Use only the simplest template (template_id=0)
    prompts = [p for p in data["prompts"] if p["template_id"] == 0]
    return prompts, data["categories"]


def run_experiment():
    """Run the full experiment: all models × all categories × N runs."""
    prompts, categories = load_prompts()
    print(f"Loaded {len(prompts)} prompts (template_id=0 only)")
    print(f"Models: {list(MODELS.keys())}")
    print(f"Runs per model per category: {NUM_RUNS}")
    print(f"Total API calls: {len(prompts) * len(MODELS) * NUM_RUNS}")

    all_results = []
    checkpoint_file = RESULTS_DIR / "responses_checkpoint.jsonl"

    # Resume from checkpoint if exists
    existing = set()
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            for line in f:
                r = json.loads(line)
                all_results.append(r)
                key = f"{r['model']}|{r['prompt']}|{r['run_id']}"
                existing.add(key)
        print(f"Resuming from checkpoint: {len(all_results)} existing results")

    total = len(prompts) * len(MODELS) * NUM_RUNS
    done = len(existing)

    for model_name in MODELS:
        print(f"\n--- Model: {model_name} ---")
        for prompt_data in prompts:
            for run_id in range(NUM_RUNS):
                key = f"{model_name}|{prompt_data['prompt']}|{run_id}"
                if key in existing:
                    continue

                result = query_model(model_name, prompt_data["prompt"], run_id)
                result["category"] = prompt_data["category"]
                result["domain"] = prompt_data["domain"]
                result["difficulty"] = prompt_data["difficulty"]
                result["prompt_id"] = prompt_data["id"]

                all_results.append(result)

                # Append to checkpoint
                with open(checkpoint_file, "a") as f:
                    f.write(json.dumps(result) + "\n")

                done += 1
                if done % 50 == 0:
                    print(f"  Progress: {done}/{total} ({100*done/total:.1f}%)")

                # Small delay to avoid rate limits
                time.sleep(0.3)

    # Save final results
    output_file = RESULTS_DIR / "all_responses.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nDone! {len(all_results)} responses saved to {output_file}")
    return all_results


if __name__ == "__main__":
    run_experiment()
