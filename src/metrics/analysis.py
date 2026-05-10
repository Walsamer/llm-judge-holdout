"""Extended bias metrics for diagnostic analysis.

Four metrics computed from the same votes.json:

1. Wataoka (individual vote level)
   P(self-judge agrees with human | human prefers own output) - P(non-self agrees | same)
   Measures whether self-preference bias EXISTS in individual votes.

2. Jury accuracy
   P(jury majority == human) per model per condition.
   Measures whether holdout jury is more accurate overall.

3. Self-vote deviation
   For each matchup, compare self-judge's vote to other judges' majority.
   Measures how often the self-judge dissents from the panel in its own favor.

4. Score inflation
   Difference between self-judge preference and holdout jury preference.
   Measures how much the self-judge inflates its own score relative to peers.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.models.config import ARENA_IDS

if TYPE_CHECKING:
    from src.judging.jury import MatchupResult


# ── Metric 1: Wataoka individual vote level ───────────────────────────────────

@dataclass
class WataokaScore:
    model_id: str
    p_self_agrees: float    # P(Y'=1 | S=1, Y=1)
    p_other_agrees: float   # P(Y'=1 | S=0, Y=1)
    bias: float             # p_self - p_other
    n_self: int
    n_other: int


def compute_wataoka(results: list[MatchupResult]) -> list[WataokaScore]:
    """Raw Wataoka metric on individual votes — does self-preference exist at all?"""
    counts: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: {"self": [0, 0], "other": [0, 0]}
    )

    for r in results:
        for model_id, side in [(r.model_a, "A"), (r.model_b, "B")]:
            human_prefers = (
                (side == "A" and r.human_winner == "model_a") or
                (side == "B" and r.human_winner == "model_b")
            )
            if not human_prefers:
                continue

            for v in r.votes:
                if v.position_swapped:
                    continue
                y_prime = 1 if v.preference == side else 0
                bucket = "self" if v.judge_model_id == model_id else "other"
                counts[model_id][bucket][0] += y_prime
                counts[model_id][bucket][1] += 1

    scores = []
    for model_id in sorted(ARENA_IDS):
        c = counts[model_id]
        p_self = c["self"][0] / c["self"][1] if c["self"][1] > 0 else float("nan")
        p_other = c["other"][0] / c["other"][1] if c["other"][1] > 0 else float("nan")
        bias = p_self - p_other if (p_self == p_self and p_other == p_other) else float("nan")
        scores.append(WataokaScore(model_id, p_self, p_other, bias, c["self"][1], c["other"][1]))
    return scores


# ── Metric 2: Jury accuracy ───────────────────────────────────────────────────

@dataclass
class AccuracyScore:
    model_id: str
    inclusive_accuracy: float
    holdout_accuracy: float
    delta: float   # holdout - inclusive (positive = holdout is better)
    n: int


def compute_jury_accuracy(results: list[MatchupResult]) -> list[AccuracyScore]:
    inc_correct: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    hol_correct: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    for r in results:
        for model_id, side in [(r.model_a, "A"), (r.model_b, "B")]:
            votes_orig = [v for v in r.votes if not v.position_swapped]
            if not votes_orig:
                continue

            inc_verdict = _majority(votes_orig)
            hol_votes = [v for v in votes_orig if v.judge_model_id != model_id]
            hol_verdict = _majority(hol_votes) if hol_votes else None

            human_side = "A" if r.human_winner == "model_a" else ("B" if r.human_winner == "model_b" else "tie")

            inc_correct[model_id][0] += int(inc_verdict == human_side)
            inc_correct[model_id][1] += 1
            if hol_verdict is not None:
                hol_correct[model_id][0] += int(hol_verdict == human_side)
                hol_correct[model_id][1] += 1

    scores = []
    for model_id in sorted(ARENA_IDS):
        ni = inc_correct[model_id][1]
        nh = hol_correct[model_id][1]
        inc_acc = inc_correct[model_id][0] / ni if ni > 0 else float("nan")
        hol_acc = hol_correct[model_id][0] / nh if nh > 0 else float("nan")
        delta = hol_acc - inc_acc if (inc_acc == inc_acc and hol_acc == hol_acc) else float("nan")
        scores.append(AccuracyScore(model_id, inc_acc, hol_acc, delta, ni))
    return scores


# ── Metric 3: Self-vote deviation ─────────────────────────────────────────────

@dataclass
class DeviationScore:
    model_id: str
    self_favor_rate: float  # how often self-judge votes for own output when others disagree
    n_matchups: int


def compute_self_deviation(results: list[MatchupResult]) -> list[DeviationScore]:
    """How often does the self-judge vote for its own output against the panel majority?"""
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    for r in results:
        for model_id, side in [(r.model_a, "A"), (r.model_b, "B")]:
            votes_orig = [v for v in r.votes if not v.position_swapped]
            self_votes = [v for v in votes_orig if v.judge_model_id == model_id]
            other_votes = [v for v in votes_orig if v.judge_model_id != model_id]

            if not self_votes or not other_votes:
                continue

            self_pref = self_votes[0].preference
            other_majority = _majority(other_votes)

            # does self-judge favor own output more than panel?
            self_favors_own = self_pref == side
            others_favor_own = other_majority == side

            counts[model_id][1] += 1
            if self_favors_own and not others_favor_own:
                counts[model_id][0] += 1  # self-judge uniquely favors own output

    scores = []
    for model_id in sorted(ARENA_IDS):
        n = counts[model_id][1]
        rate = counts[model_id][0] / n if n > 0 else float("nan")
        scores.append(DeviationScore(model_id, rate, n))
    return scores


# ── Metric 4: Score inflation ─────────────────────────────────────────────────

@dataclass
class InflationScore:
    model_id: str
    mean_self_inflation: float  # avg difference: self_vote_for_own - holdout_vote_rate_for_own
    n: int


def compute_score_inflation(results: list[MatchupResult]) -> list[InflationScore]:
    """Per-matchup: self-judge preference for own output minus holdout jury rate."""
    totals: dict[str, list] = defaultdict(lambda: [0.0, 0])

    for r in results:
        for model_id, side in [(r.model_a, "A"), (r.model_b, "B")]:
            votes_orig = [v for v in r.votes if not v.position_swapped]
            self_votes = [v for v in votes_orig if v.judge_model_id == model_id]
            other_votes = [v for v in votes_orig if v.judge_model_id != model_id]

            if not self_votes or not other_votes:
                continue

            self_score = 1.0 if self_votes[0].preference == side else 0.0
            other_rate = sum(1 for v in other_votes if v.preference == side) / len(other_votes)

            totals[model_id][0] += self_score - other_rate
            totals[model_id][1] += 1

    scores = []
    for model_id in sorted(ARENA_IDS):
        n = totals[model_id][1]
        mean = totals[model_id][0] / n if n > 0 else float("nan")
        scores.append(InflationScore(model_id, mean, n))
    return scores


# ── Helper ────────────────────────────────────────────────────────────────────

def _majority(votes) -> str:
    """Jury verdict: 'A' if A-votes > B-votes, 'B' if B > A, 'tie' if equal.

    Individual 'tie' votes abstain and are not counted for either side.
    Consistent with bias.py's n_for > n_against rule.
    """
    if not votes:
        return "tie"
    n_a = sum(1 for v in votes if v.preference == "A")
    n_b = sum(1 for v in votes if v.preference == "B")
    if n_a > n_b:
        return "A"
    if n_b > n_a:
        return "B"
    return "tie"
