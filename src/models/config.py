from dataclasses import dataclass

@dataclass(frozen=True)
class ModelConfig:
    name: str           # human-readable name for logging/tables
    arena_id: str       # model ID as it appears in the Arena dataset
    openrouter_id: str  # model slug for the OpenRouter API
    family: str         # training lineage


MODEL_POOL: list[ModelConfig] = [
    ModelConfig(
        name="GPT-4.1",
        arena_id="gpt-4.1-2025-04-14",
        openrouter_id="openai/gpt-4.1",
        family="OpenAI",
    ),
    ModelConfig(
        name="Claude Opus 4",
        arena_id="claude-opus-4-20250514",
        openrouter_id="anthropic/claude-opus-4",
        family="Anthropic",
    ),
    ModelConfig(
        name="Gemini 2.5 Pro",
        arena_id="gemini-2.5-pro",
        openrouter_id="google/gemini-2.5-pro",
        family="Google",
    ),
    ModelConfig(
        name="Grok 3",
        arena_id="grok-3-preview-02-24",
        openrouter_id="x-ai/grok-3",
        family="xAI",
    ),
    ModelConfig(
        name="DeepSeek V3",
        arena_id="deepseek-v3-0324",
        openrouter_id="deepseek/deepseek-chat-v3-0324",
        family="DeepSeek",
    ),
    ModelConfig(
        name="Llama 4 Maverick",
        arena_id="llama-4-maverick-03-26-experimental",
        openrouter_id="meta-llama/llama-4-maverick:floor",
        family="Meta",
    ),
    ModelConfig(
        name="Mistral Medium",
        arena_id="mistral-medium-2505",
        openrouter_id="mistralai/mistral-medium-3",
        family="Mistral",
    ),
]

MODEL_BY_ARENA_ID: dict[str, ModelConfig] = {m.arena_id: m for m in MODEL_POOL}
ARENA_IDS: set[str] = {m.arena_id for m in MODEL_POOL}
