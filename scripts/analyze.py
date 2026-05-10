#!/usr/bin/env python3
"""Run all four diagnostic metrics on saved votes.json and print a full report."""
import sys
import json
from dataclasses import dataclass
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.metrics.analysis import (
    compute_wataoka,
    compute_jury_accuracy,
    compute_self_deviation,
    compute_score_inflation,
)

VOTES_PATH = Path(__file__).parents[1] / "results" / "votes.json"


@dataclass
class JudgeVote:
    judge_model_id: str
    generator_model_id: str
    preference: str
    position_swapped: bool


@dataclass
class MatchupResult:
    matchup_id: str
    model_a: str
    model_b: str
    human_winner: str
    votes: list[JudgeVote]


def load_votes(path: Path) -> list[MatchupResult]:
    """Load saved votes without importing API client dependencies."""
    raw = json.loads(path.read_text())
    return [
        MatchupResult(
            matchup_id=r["matchup_id"],
            model_a=r["model_a"],
            model_b=r["model_b"],
            human_winner=r["human_winner"],
            votes=[JudgeVote(**v) for v in r["votes"]],
        )
        for r in raw
    ]


def main():
    results = load_votes(VOTES_PATH)
    print(f"Loaded {len(results)} matchups\n")

    # ── Metric 1: Wataoka ────────────────────────────────────────────────────
    print("=" * 70)
    print("METRIC 1 — Wataoka (individual vote level)")
    print("P(self agrees with human | human prefers own) - P(other agrees | same)")
    print("Positive = self-judge inflates own score vs other judges")
    print("-" * 70)
    print(f"{'Model':<40} {'P(self)':>8} {'P(other)':>9} {'Bias':>8} {'N_self':>7}")
    for s in compute_wataoka(results):
        print(f"{s.model_id:<40} {s.p_self_agrees:>8.4f} {s.p_other_agrees:>9.4f} {s.bias:>8.4f} {s.n_self:>7}")

    # ── Metric 2: Jury accuracy ──────────────────────────────────────────────
    print()
    print("=" * 70)
    print("METRIC 2 — Jury accuracy vs human preference")
    print("Delta = holdout - inclusive (positive = holdout jury is more accurate)")
    print("-" * 70)
    print(f"{'Model':<40} {'Inclusive':>10} {'Holdout':>9} {'Delta':>7} {'N':>5}")
    for s in compute_jury_accuracy(results):
        print(f"{s.model_id:<40} {s.inclusive_accuracy:>10.4f} {s.holdout_accuracy:>9.4f} {s.delta:>7.4f} {s.n:>5}")

    # ── Metric 3: Self-vote deviation ────────────────────────────────────────
    print()
    print("=" * 70)
    print("METRIC 3 — Self-vote deviation")
    print("Rate at which self-judge uniquely votes for own output (panel disagrees)")
    print("-" * 70)
    print(f"{'Model':<40} {'Self-favor rate':>16} {'N matchups':>11}")
    for s in compute_self_deviation(results):
        print(f"{s.model_id:<40} {s.self_favor_rate:>16.4f} {s.n_matchups:>11}")

    # ── Metric 4: Score inflation ────────────────────────────────────────────
    print()
    print("=" * 70)
    print("METRIC 4 — Score inflation")
    print("Mean difference: self-judge vote for own output minus holdout jury rate")
    print("Positive = self-judge is more favorable to own output than peers")
    print("-" * 70)
    print(f"{'Model':<40} {'Mean inflation':>15} {'N':>5}")
    for s in compute_score_inflation(results):
        print(f"{s.model_id:<40} {s.mean_self_inflation:>15.4f} {s.n:>5}")


if __name__ == "__main__":
    main()
