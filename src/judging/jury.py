"""Jury scoring with position-swapping and concurrent API calls.

Run once — inclusive (all 7 judges). Holdout is derived mathematically
in the bias metric by filtering out self-votes, so no second run needed.

Concurrency: the 14 calls per matchup (7 judges × 2 positions) are
dispatched in parallel via ThreadPoolExecutor. Sequential across matchups
so incremental saving works correctly.
"""
from __future__ import annotations
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path

from src.data.arena import Matchup
from src.models.config import ModelConfig, MODEL_POOL
from src.models.client import judge

MAX_WORKERS = 14  # 7 judges × 2 positions — all independent


@dataclass
class JudgeVote:
    judge_model_id: str
    generator_model_id: str  # stored as matchup's model_a for historical reasons;
                              # field is not used by any metric — do not rely on it
    preference: str           # "A" | "B" | "tie"
    position_swapped: bool


@dataclass
class MatchupResult:
    matchup_id: str
    model_a: str
    model_b: str
    human_winner: str
    votes: list[JudgeVote] = field(default_factory=list)


def run_jury(
    matchups: list[Matchup],
    models: list[ModelConfig] = MODEL_POOL,
    save_path: Path | None = None,
) -> list[MatchupResult]:
    """Judge all matchups with all models, saving incrementally."""
    existing = _load_existing(save_path)
    done_ids = {r.matchup_id for r in existing}
    results = list(existing)

    todo = [m for m in matchups if m.matchup_id not in done_ids]
    print(f"  {len(existing)} already done, {len(todo)} remaining")

    for i, m in enumerate(todo):
        result = _judge_matchup(m, models)
        results.append(result)
        if save_path:
            _save(results, save_path)
        print(f"  [{i+1}/{len(todo)}] {m.matchup_id[:8]}… done")

    return results


def _judge_matchup(m: Matchup, models: list[ModelConfig]) -> MatchupResult:
    """Dispatch all 14 judge calls for one matchup concurrently."""
    result = MatchupResult(m.matchup_id, m.model_a, m.model_b, m.human_winner)

    tasks = [(model, swapped) for model in models for swapped in (False, True)]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(_call, model, m, swapped): (model, swapped)
            for model, swapped in tasks
        }
        for future in as_completed(futures):
            model, swapped = futures[future]
            pref = future.result()
            if pref is None:
                continue  # failed call — skip rather than storing bad data
            result.votes.append(JudgeVote(
                judge_model_id=model.arena_id,
                generator_model_id=m.model_a,
                preference=pref,
                position_swapped=swapped,
            ))

    return result


def _call(model: ModelConfig, m: Matchup, swapped: bool) -> str:
    if swapped:
        pref = judge(model, m.prompt, m.response_b, m.response_a)
        return _flip(pref)
    return judge(model, m.prompt, m.response_a, m.response_b)


def _flip(pref: str) -> str:
    return {"A": "B", "B": "A"}.get(pref, pref)


def _save(results: list[MatchupResult], path: Path) -> None:
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps([_result_to_dict(r) for r in results], indent=2))


def _load_existing(path: Path | None) -> list[MatchupResult]:
    if path is None or not path.exists():
        return []
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


def _result_to_dict(r: MatchupResult) -> dict:
    return {
        "matchup_id": r.matchup_id,
        "model_a": r.model_a,
        "model_b": r.model_b,
        "human_winner": r.human_winner,
        "votes": [asdict(v) for v in r.votes],
    }
