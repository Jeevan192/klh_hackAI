import json
import os
from pathlib import Path

from openai import OpenAI


SYSTEM_PROMPT = (
    "You are a recruitment fraud analyst. Given structured fraud signals, "
    "write a 3-sentence recruiter alert. Be factual and specific. "
    "Do not assign blame. Do not say the person is definitely fraudulent. "
    "Use the signal data provided."
)


def _load_env_file() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return

    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception:
        pass


_load_env_file()


def generate_explanation(signals_dict: dict, risk_score: int, risk_level: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "Explanation unavailable - review signals manually."

    payload = {
        "signals": signals_dict,
        "risk_score": risk_score,
        "risk_level": risk_level,
    }

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload)},
            ],
            temperature=0.2,
        )
        content = response.choices[0].message.content if response.choices else None
        if not content:
            return "Explanation unavailable - review signals manually."
        return content.strip()
    except Exception:
        return "Explanation unavailable - review signals manually."
