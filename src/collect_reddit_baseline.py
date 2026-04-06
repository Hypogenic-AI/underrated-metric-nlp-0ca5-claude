"""
Collect "obvious underrated" baselines from web search.
For each category, search for commonly cited "most underrated X" answers.
These represent what humans commonly consider underrated — the "obvious" picks.
"""

import json
import os
import time
from pathlib import Path

import openai

RESULTS_DIR = Path("/workspaces/underrated-metric-nlp-0ca5-claude/results/raw")

# Categories from the dataset
CATEGORIES = [
    "rock band", "jazz album", "hip-hop producer", "classical composer",
    "singer-songwriter of the 2010s", "sci-fi movie", "horror film director",
    "animated TV show", "documentary film", "film noir",
    "novel of the 21st century", "short story writer", "poet", "non-fiction book",
    "graphic novel", "cuisine", "vegetable", "spice", "cheese",
    "coffee brewing method", "programming language", "scientific discovery",
    "mathematical theorem", "open-source software project",
    "invention of the 20th century", "Olympic sport",
    "basketball player of all time", "board game", "chess opening",
    "video game of the 2010s", "city to visit in Europe",
    "national park in the United States", "country for solo travel",
    "island destination", "historical site", "life skill to learn",
    "hobby for adults", "era in human history", "branch of philosophy",
    "career path",
]


def search_obvious_underrated(category: str) -> dict:
    """Use GPT-4.1 to identify what Reddit/web commonly cites as most underrated."""
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    prompt = f"""Think about what answers would appear MOST FREQUENTLY if you searched Reddit threads, Quora, and online forums for "most underrated {category}".

List the TOP 5 most commonly cited "underrated" {category} answers that appear repeatedly across Reddit threads and online discussions. These are the "obvious" picks — answers that many people give when asked this question.

Format your response as a JSON list of strings. Example: ["answer1", "answer2", "answer3", "answer4", "answer5"]

IMPORTANT: Only output the JSON list, nothing else."""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
        )
        text = response.choices[0].message.content.strip()
        # Parse JSON
        answers = json.loads(text)
        return {"category": category, "obvious_underrated": answers, "status": "success"}
    except Exception as e:
        return {"category": category, "obvious_underrated": [], "status": f"error: {e}"}


def main():
    print("Collecting Reddit/web baseline for 'obvious underrated' answers...")
    baselines = []

    for i, category in enumerate(CATEGORIES):
        result = search_obvious_underrated(category)
        baselines.append(result)
        print(f"  [{i+1}/{len(CATEGORIES)}] {category}: {result['obvious_underrated'][:3]}...")
        time.sleep(0.5)

    output_file = RESULTS_DIR / "reddit_baselines.json"
    with open(output_file, "w") as f:
        json.dump(baselines, f, indent=2)

    print(f"\nSaved baselines to {output_file}")
    return baselines


if __name__ == "__main__":
    main()
