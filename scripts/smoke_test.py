#!/usr/bin/env python3
"""Verify the full pipeline works end-to-end with a single matchup."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.models.config import MODEL_POOL
from src.models.client import judge

PROMPT = "Explain what a neural network is in one sentence."
RESPONSE_A = "A neural network is a computational model inspired by the brain that learns patterns from data through layers of interconnected nodes."
RESPONSE_B = "Its a thing that does machine learning stuff with lots of numbers."

def main():
    print("Smoke test — calling each model once...\n")
    all_ok = True
    for model in MODEL_POOL:
        try:
            result = judge(model, PROMPT, RESPONSE_A, RESPONSE_B)
            print(f"  {model.name:<22} → {result}")
        except Exception as e:
            print(f"  {model.name:<22} → FAILED: {e}")
            all_ok = False

    print()
    if all_ok:
        print("All models OK. Ready to run the experiment.")
    else:
        print("Some models failed. Check your API key and OpenRouter model slugs.")

if __name__ == "__main__":
    main()
