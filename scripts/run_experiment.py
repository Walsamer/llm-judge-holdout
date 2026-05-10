#!/usr/bin/env python3
"""Main experiment runner.

Run once — saves raw votes incrementally to results/votes.json.
Bias metrics and stats are computed from the saved votes and can be
recomputed at any time without re-running API calls.
"""
import json
import sys
from dataclasses import asdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.data.arena import load_arena_matchups
from src.judging.jury import run_jury, _result_to_dict
from src.metrics.bias import compute_inclusive_bias, compute_holdout_bias
from src.metrics.stats import wilcoxon_test
from src.models.config import ARENA_IDS

RESULTS_DIR = Path(__file__).parents[1] / "results"
VOTES_PATH = RESULTS_DIR / "votes.json"


def main(max_samples: int = 300):
    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"Loading up to {max_samples} Arena matchups...")
    matchups = load_arena_matchups(ARENA_IDS, max_samples)
    print(f"  Loaded {len(matchups)} matchups")

    print("Running jury (saves incrementally to results/votes.json)...")
    results = run_jury(matchups, save_path=VOTES_PATH)
    print(f"  Done — {len(results)} matchups judged")

    _compute_and_report(results)


def recompute():
    """Recompute metrics from saved votes without calling any APIs."""
    if not VOTES_PATH.exists():
        print("No votes.json found. Run the experiment first.")
        return
    from src.judging.jury import _load_existing
    results = _load_existing(VOTES_PATH)
    print(f"Loaded {len(results)} matchups from votes.json")
    _compute_and_report(results)


def _compute_and_report(results):
    inclusive_scores = compute_inclusive_bias(results)
    holdout_scores = compute_holdout_bias(results)

    print("\n=== False Endorsement Rate (lower = less biased) ===")
    print(f"{'Model':<40} {'Inclusive':>10} {'Holdout':>10} {'Delta':>10}")
    inc_by_id = {s.model_id: s for s in inclusive_scores}
    hol_by_id = {s.model_id: s for s in holdout_scores}
    for model_id in sorted(inc_by_id):
        inc = inc_by_id[model_id].false_endorsement_rate
        hol = hol_by_id.get(model_id)
        hol_val = hol.false_endorsement_rate if hol else float("nan")
        delta = inc - hol_val if hol_val == hol_val else float("nan")
        print(f"{model_id:<40} {inc:>10.4f} {hol_val:>10.4f} {delta:>10.4f}")

    stat = wilcoxon_test(inclusive_scores, holdout_scores)
    print(f"\nWilcoxon: stat={stat.statistic:.3f}, p={stat.p_value:.4f}, r={stat.effect_size:.3f}, n={stat.n}")
    print(f"Significant (p<0.05): {stat.significant}")

    out = {
        "inclusive_bias": [asdict(s) for s in inclusive_scores],
        "holdout_bias": [asdict(s) for s in holdout_scores],
        "wilcoxon": asdict(stat),
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(out, indent=2))
    print(f"\nMetrics saved to results/metrics.json")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--max-samples", type=int, default=300)
    p.add_argument("--recompute", action="store_true", help="Recompute metrics from saved votes")
    args = p.parse_args()
    if args.recompute:
        recompute()
    else:
        main(args.max_samples)
