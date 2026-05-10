"""Jury-level self-preference bias metric.

For each model X across all matchups where X generated one of the responses:

  false_endorsement_rate = P(jury prefers X | human preferred the other model)

Self-preference bias inflates this rate in the inclusive condition (the self-judge
votes for its own output even when the human preferred the opponent). Removing the
self-judge (holdout) should reduce it.

This is computed at the jury level (majority vote across all judges) rather than
the individual vote level, allowing direct comparison between inclusive and holdout
configurations.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict

from src.judging.jury import MatchupResult
from src.models.config import ARENA_IDS


@dataclass
class BiasScore:
    model_id: str
    false_endorsement_rate: float  # P(jury prefers X | human preferred other)
    true_endorsement_rate: float   # P(jury prefers X | human preferred X)
    n_human_preferred: int         # matchups where human preferred X
    n_human_rejected: int          # matchups where human preferred other
    condition: str                 # "inclusive" | "holdout"


def compute_inclusive_bias(results: list[MatchupResult]) -> list[BiasScore]:
    return _compute(results, exclude_self=False)


def compute_holdout_bias(results: list[MatchupResult]) -> list[BiasScore]:
    return _compute(results, exclude_self=True)


def _compute(results: list[MatchupResult], exclude_self: bool) -> list[BiasScore]:
    # counts[model_id] = {"preferred": [jury_endorsed, total], "rejected": [jury_endorsed, total]}
    counts: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: {"preferred": [0, 0], "rejected": [0, 0]}
    )

    for r in results:
        for model_id, side in [(r.model_a, "A"), (r.model_b, "B")]:
            if model_id not in ARENA_IDS:
                continue

            # filter votes for this condition
            votes = [
                v for v in r.votes
                if not (exclude_self and v.judge_model_id == model_id)
                and not v.position_swapped  # use original orientation only; swapped votes are already normalised
            ]
            if not votes:
                continue

            # jury verdict: majority of filtered votes
            n_for = sum(1 for v in votes if v.preference == side)
            n_against = sum(1 for v in votes if v.preference != side and v.preference != "tie")
            jury_endorses = n_for > n_against

            human_preferred = (
                (side == "A" and r.human_winner == "model_a") or
                (side == "B" and r.human_winner == "model_b")
            )
            human_rejected = (
                (side == "A" and r.human_winner == "model_b") or
                (side == "B" and r.human_winner == "model_a")
            )

            if human_preferred:
                counts[model_id]["preferred"][0] += int(jury_endorses)
                counts[model_id]["preferred"][1] += 1
            elif human_rejected:
                counts[model_id]["rejected"][0] += int(jury_endorses)
                counts[model_id]["rejected"][1] += 1

    condition = "holdout" if exclude_self else "inclusive"
    scores = []
    for model_id in sorted(ARENA_IDS):
        c = counts[model_id]
        fer = c["rejected"][0] / c["rejected"][1] if c["rejected"][1] > 0 else float("nan")
        ter = c["preferred"][0] / c["preferred"][1] if c["preferred"][1] > 0 else float("nan")
        scores.append(BiasScore(
            model_id=model_id,
            false_endorsement_rate=fer,
            true_endorsement_rate=ter,
            n_human_preferred=c["preferred"][1],
            n_human_rejected=c["rejected"][1],
            condition=condition,
        ))
    return scores
