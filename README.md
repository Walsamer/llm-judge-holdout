# Self-Preference Bias in LLM Juries: A Controlled Test of Hard Judge Exclusion

Samuel Eder — Johannes Kepler University Linz (JKU)

---

## Summary

LLM-as-a-Judge ensembles reduce individual judge noise by aggregating across multiple models. But when the evaluated model is included in its own jury — the standard *inclusive* configuration — self-preference bias is not eliminated: the self-serving vote is simply averaged in with the rest.

This paper asks: does *hard holdout judging* (programmatically excluding the generating model from its own jury) reduce self-preference bias on subjective open-ended evaluation tasks?

**Short answer: not at n=7.** Using 300 pairwise matchups from Chatbot Arena and a seven-model panel, we find no significant reduction in jury-level false endorsement rate ($p = 0.875$). Self-preference is clearly present at the individual-vote level (six of seven models, with Claude Opus 4 at +0.272 and Gemini 2.5 Pro at +0.254), but a diverse seven-judge panel absorbs it without requiring exclusion. The holdout benefit is real but concentrated in very small juries — most clearly at k=3 (Δ = +0.093).

## Repository Structure

```
.
├── src/
│   ├── data/arena.py          # Chatbot Arena dataset loading and sampling
│   ├── judging/jury.py        # Jury construction and vote aggregation
│   ├── metrics/bias.py        # Wataoka Equal Opportunity metric
│   ├── metrics/stats.py       # Wilcoxon test, rank-biserial correlation
│   └── models/client.py       # OpenRouter API client
│
├── scripts/
│   ├── run_experiment.py      # Main experiment: collect all judge votes
│   ├── analyze.py             # Compute FER, Wataoka bias, and diagnostics
│   ├── simulate_jury_size.py  # Jury-size sweep (k=2..7) via resampling
│   ├── theoretical_jury_size.py  # Pivotal-voter formula derivation
│   ├── explore_arena.py       # Dataset exploration utilities
│   └── smoke_test.py          # Sanity check before a full run
│
└── requirements.txt
```

## Dataset

The 300 sampled matchups and 4,196 individual judge votes are published separately on HuggingFace:

**[huggingface.co/datasets/clarsam/llm-judge-holdout](https://huggingface.co/datasets/clarsam/llm-judge-holdout)**

| File | Description |
|------|-------------|
| `arena_matchups.json` | 300 pairwise matchups: prompt, both responses, human winner |
| `votes.json` | 4,196 individual judge votes across all 7 models × 2 positions |

Download both files into `data/` and `results/` respectively to run the analysis scripts.

## Reproducing the Experiment

### Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# add your OPENROUTER_API_KEY to .env
```

### Run order

```bash
# 1. Smoke test with a small sample
python scripts/smoke_test.py

# 2. Full experiment — collects all judge votes (~$37, ~45 min)
python scripts/run_experiment.py

# 3. Compute all metrics from votes.json
python scripts/analyze.py

# 4. Jury-size simulation
python scripts/simulate_jury_size.py

# 5. Theoretical pivotal-voter model
python scripts/theoretical_jury_size.py
```

Steps 3–5 only require the downloaded `votes.json` and do not call any APIs.

## Model Pool

| Model | Provider |
|-------|----------|
| GPT-4.1 | OpenAI |
| Claude Opus 4 | Anthropic |
| Gemini 2.5 Pro | Google |
| Grok 3 | xAI |
| DeepSeek V3 | DeepSeek |
| Llama 4 Maverick | Meta |
| Mistral Medium | Mistral |

All API calls routed through [OpenRouter](https://openrouter.ai).

## Key Results

| Metric | Value |
|--------|-------|
| Matchups | 300 |
| Total judge votes | 4,196 |
| Wilcoxon $p$ (H1: holdout reduces FER) | 0.875 — **null result** |
| Models with positive Wataoka bias | 6 / 7 |
| Holdout benefit at k=3 | Δ = +0.093 (95% CI [+0.062, +0.128]) |

## Citation

```bibtex
@article{eder2026holdout,
  title   = {Self-Preference Bias in LLM Juries: A Controlled Test of Hard Judge Exclusion},
  author  = {Eder, Samuel},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## License

Code: MIT. Dataset: CC BY 4.0 (derived from [lmarena-ai/arena-human-preference-140k](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k)).
