"""Single OpenRouter client for all judge calls."""
import os
import re
import time
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
from .config import ModelConfig

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

JUDGE_SYSTEM_PROMPT = (
    "You are an impartial judge evaluating the quality of two AI assistant responses. "
    "Given a user prompt and two responses (Response A and Response B), decide which is better. "
    "Reply with exactly one word: A, B, or tie."
)


def judge(model: ModelConfig, prompt: str, response_a: str, response_b: str) -> str | None:
    """Returns A/B/tie, or None if all retries fail."""
    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
    )
    user_content = (
        f"User prompt:\n{prompt}\n\n"
        f"Response A:\n{response_a}\n\n"
        f"Response B:\n{response_b}"
    )
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model.openrouter_id,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=200,
                temperature=0,
            )
            return _parse_preference(resp.choices[0].message.content)
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                print(f"    [WARN] {model.name} failed after {MAX_RETRIES} attempts: {e}")
                return None


def _parse_preference(text: str) -> str:
    text = text.strip().lower()
    if re.search(r"\btie\b", text):
        return "tie"
    if re.search(r"\b(response\s*)?a\b", text):
        return "A"
    if re.search(r"\b(response\s*)?b\b", text):
        return "B"
    return "tie"
