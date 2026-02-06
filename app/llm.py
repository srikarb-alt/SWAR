import time
import json
import requests
from app.config import GEMINI_API_KEY, MODEL, LLM_PROMPT

MAX_RETRIES = 3
RETRY_DELAY = 30


def call_gemini(gt, asr):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={GEMINI_API_KEY}"

    payload = {
        "system_instruction": {"parts": [{"text": LLM_PROMPT}]},
        "contents": [{"parts": [{"text": f"GT:\n{gt}\n\nASR:\n{asr}"}]}]
    }

    for attempt in range(MAX_RETRIES):
        try:
            res = requests.post(url, json=payload, timeout=60)
            data = res.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            text = text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)

    return None
