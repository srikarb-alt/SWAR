import json
import random
import re
import time

import requests

from app.config import GEMINI_API_KEY, MODEL, LLM_PROMPT

MAX_RETRIES = 3
RETRY_DELAY = 30


def _extract_candidate_text(data):
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as exc:
        raise ValueError("Missing candidate content in Gemini response") from exc


def _compute_backoff_delay(
    base_delay,
    attempt,
    backoff_factor,
    max_retry_delay,
    jitter_ratio,
):
    raw_delay = min(max_retry_delay, base_delay * (backoff_factor ** attempt))
    jitter_window = max(0.0, raw_delay * jitter_ratio)
    return max(0.0, raw_delay + random.uniform(-jitter_window, jitter_window))


def _is_retryable_status(status_code):
    return status_code in {408, 429, 500, 502, 503, 504}


def _has_minimum_fields(parsed):
    if not isinstance(parsed, dict):
        return False
    return all(key in parsed for key in ("gt_translit", "asr_translit", "GT_Tokens"))


def call_gemini_with_details(
    gt,
    asr,
    max_retries=MAX_RETRIES,
    retry_delay=RETRY_DELAY,
    backoff_factor=2.0,
    max_retry_delay=90,
    jitter_ratio=0.2,
):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={GEMINI_API_KEY}"

    payload = {
        "system_instruction": {"parts": [{"text": LLM_PROMPT}]},
        "contents": [{"parts": [{"text": f"GT:\n{gt}\n\nASR:\n{asr}"}]}],
    }

    errors = []

    for attempt in range(max_retries):
        try:
            res = requests.post(url, json=payload, timeout=60)
            res.raise_for_status()
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response else None
            errors.append(f"attempt {attempt + 1}: request failed (status {status_code})")
            if status_code is not None and not _is_retryable_status(status_code):
                return None, "; ".join(errors)
            if attempt < max_retries - 1:
                sleep_for = _compute_backoff_delay(
                    base_delay=retry_delay,
                    attempt=attempt,
                    backoff_factor=backoff_factor,
                    max_retry_delay=max_retry_delay,
                    jitter_ratio=jitter_ratio,
                )
                time.sleep(sleep_for)
            continue
        except requests.RequestException as exc:
            errors.append(f"attempt {attempt + 1}: request failed ({exc})")
            if attempt < max_retries - 1:
                sleep_for = _compute_backoff_delay(
                    base_delay=retry_delay,
                    attempt=attempt,
                    backoff_factor=backoff_factor,
                    max_retry_delay=max_retry_delay,
                    jitter_ratio=jitter_ratio,
                )
                time.sleep(sleep_for)
            continue

        try:
            data = res.json()
        except ValueError as exc:
            errors.append(f"attempt {attempt + 1}: response is not valid JSON ({exc})")
            if attempt < max_retries - 1:
                sleep_for = _compute_backoff_delay(
                    base_delay=retry_delay,
                    attempt=attempt,
                    backoff_factor=backoff_factor,
                    max_retry_delay=max_retry_delay,
                    jitter_ratio=jitter_ratio,
                )
                time.sleep(sleep_for)
            continue

        try:
            text = _extract_candidate_text(data)
        except ValueError as exc:
            errors.append(f"attempt {attempt + 1}: {exc}")
            if attempt < max_retries - 1:
                sleep_for = _compute_backoff_delay(
                    base_delay=retry_delay,
                    attempt=attempt,
                    backoff_factor=backoff_factor,
                    max_retry_delay=max_retry_delay,
                    jitter_ratio=jitter_ratio,
                )
                time.sleep(sleep_for)
            continue

        cleaned = text.replace("```json", "").replace("```", "").strip()
        if not cleaned:
            errors.append(f"attempt {attempt + 1}: empty model response")
            if attempt < max_retries - 1:
                sleep_for = _compute_backoff_delay(
                    base_delay=retry_delay,
                    attempt=attempt,
                    backoff_factor=backoff_factor,
                    max_retry_delay=max_retry_delay,
                    jitter_ratio=jitter_ratio,
                )
                time.sleep(sleep_for)
            continue

        parsed = None
        parse_error = None
        for candidate in (cleaned,):
            try:
                parsed = json.loads(candidate)
                break
            except json.JSONDecodeError as exc:
                parse_error = exc
        if parsed is None:
            match = re.search(r"\{[\s\S]*\}", cleaned)
            if match:
                try:
                    parsed = json.loads(match.group())
                except json.JSONDecodeError as exc:
                    parse_error = exc

        if parsed is None:
            errors.append(f"attempt {attempt + 1}: model output not valid JSON ({parse_error})")
            if attempt < max_retries - 1:
                sleep_for = _compute_backoff_delay(
                    base_delay=retry_delay,
                    attempt=attempt,
                    backoff_factor=backoff_factor,
                    max_retry_delay=max_retry_delay,
                    jitter_ratio=jitter_ratio,
                )
                time.sleep(sleep_for)
            continue

        if not _has_minimum_fields(parsed):
            errors.append(f"attempt {attempt + 1}: model output missing required fields")
            if attempt < max_retries - 1:
                sleep_for = _compute_backoff_delay(
                    base_delay=retry_delay,
                    attempt=attempt,
                    backoff_factor=backoff_factor,
                    max_retry_delay=max_retry_delay,
                    jitter_ratio=jitter_ratio,
                )
                time.sleep(sleep_for)
            continue

        return parsed, None

    return None, "; ".join(errors) if errors else "unknown error"


def call_gemini(gt, asr):
    parsed, _ = call_gemini_with_details(
        gt,
        asr,
        max_retries=MAX_RETRIES,
        retry_delay=RETRY_DELAY,
        backoff_factor=1.0,
        max_retry_delay=RETRY_DELAY,
        jitter_ratio=0.0,
    )
    return parsed


def _build_batch_user_text(items):
    lines = [
        "Return JSON ONLY.",
        "You will receive multiple items. Return a JSON array with the same length and order.",
        "Each array element must follow the exact schema required by the system instruction.",
        "",
    ]
    for index, item in enumerate(items, start=1):
        lines.append(f"Item {index}:")
        lines.append(f"GT: {item['gt']}")
        lines.append(f"ASR: {item['asr']}")
        lines.append("")
    return "\n".join(lines).strip()


def call_gemini_batch_with_details(
    items,
    max_retries=MAX_RETRIES,
    retry_delay=RETRY_DELAY,
    backoff_factor=2.0,
    max_retry_delay=90,
    jitter_ratio=0.2,
):
    if not items:
        return [], None

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={GEMINI_API_KEY}"
    user_text = _build_batch_user_text(items)

    payload = {
        "system_instruction": {"parts": [{"text": LLM_PROMPT}]},
        "contents": [{"parts": [{"text": user_text}]}],
    }

    errors = []

    for attempt in range(max_retries):
        try:
            res = requests.post(url, json=payload, timeout=90)
            res.raise_for_status()
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response else None
            errors.append(f"attempt {attempt + 1}: request failed (status {status_code})")
            if status_code is not None and not _is_retryable_status(status_code):
                return None, "; ".join(errors)
            if attempt < max_retries - 1:
                sleep_for = _compute_backoff_delay(
                    base_delay=retry_delay,
                    attempt=attempt,
                    backoff_factor=backoff_factor,
                    max_retry_delay=max_retry_delay,
                    jitter_ratio=jitter_ratio,
                )
                time.sleep(sleep_for)
            continue
        except requests.RequestException as exc:
            errors.append(f"attempt {attempt + 1}: request failed ({exc})")
            if attempt < max_retries - 1:
                sleep_for = _compute_backoff_delay(
                    base_delay=retry_delay,
                    attempt=attempt,
                    backoff_factor=backoff_factor,
                    max_retry_delay=max_retry_delay,
                    jitter_ratio=jitter_ratio,
                )
                time.sleep(sleep_for)
            continue

        try:
            data = res.json()
        except ValueError as exc:
            errors.append(f"attempt {attempt + 1}: response is not valid JSON ({exc})")
            if attempt < max_retries - 1:
                sleep_for = _compute_backoff_delay(
                    base_delay=retry_delay,
                    attempt=attempt,
                    backoff_factor=backoff_factor,
                    max_retry_delay=max_retry_delay,
                    jitter_ratio=jitter_ratio,
                )
                time.sleep(sleep_for)
            continue

        try:
            text = _extract_candidate_text(data)
        except ValueError as exc:
            errors.append(f"attempt {attempt + 1}: {exc}")
            if attempt < max_retries - 1:
                sleep_for = _compute_backoff_delay(
                    base_delay=retry_delay,
                    attempt=attempt,
                    backoff_factor=backoff_factor,
                    max_retry_delay=max_retry_delay,
                    jitter_ratio=jitter_ratio,
                )
                time.sleep(sleep_for)
            continue

        cleaned = text.replace("```json", "").replace("```", "").strip()
        if not cleaned:
            errors.append(f"attempt {attempt + 1}: empty model response")
            if attempt < max_retries - 1:
                sleep_for = _compute_backoff_delay(
                    base_delay=retry_delay,
                    attempt=attempt,
                    backoff_factor=backoff_factor,
                    max_retry_delay=max_retry_delay,
                    jitter_ratio=jitter_ratio,
                )
                time.sleep(sleep_for)
            continue

        parsed = None
        parse_error = None
        for candidate in (cleaned,):
            try:
                parsed = json.loads(candidate)
                break
            except json.JSONDecodeError as exc:
                parse_error = exc

        if parsed is None:
            match = re.search(r"\[[\s\S]*\]", cleaned)
            if match:
                try:
                    parsed = json.loads(match.group())
                except json.JSONDecodeError as exc:
                    parse_error = exc

        if parsed is None:
            errors.append(f"attempt {attempt + 1}: model output not valid JSON ({parse_error})")
            if attempt < max_retries - 1:
                sleep_for = _compute_backoff_delay(
                    base_delay=retry_delay,
                    attempt=attempt,
                    backoff_factor=backoff_factor,
                    max_retry_delay=max_retry_delay,
                    jitter_ratio=jitter_ratio,
                )
                time.sleep(sleep_for)
            continue

        if not isinstance(parsed, list):
            errors.append(f"attempt {attempt + 1}: model output is not a JSON array")
            if attempt < max_retries - 1:
                sleep_for = _compute_backoff_delay(
                    base_delay=retry_delay,
                    attempt=attempt,
                    backoff_factor=backoff_factor,
                    max_retry_delay=max_retry_delay,
                    jitter_ratio=jitter_ratio,
                )
                time.sleep(sleep_for)
            continue

        if len(parsed) != len(items):
            errors.append(
                f"attempt {attempt + 1}: expected {len(items)} results, got {len(parsed)}"
            )
            if attempt < max_retries - 1:
                sleep_for = _compute_backoff_delay(
                    base_delay=retry_delay,
                    attempt=attempt,
                    backoff_factor=backoff_factor,
                    max_retry_delay=max_retry_delay,
                    jitter_ratio=jitter_ratio,
                )
                time.sleep(sleep_for)
            continue

        if not all(_has_minimum_fields(item) for item in parsed):
            errors.append(f"attempt {attempt + 1}: model output missing required fields")
            if attempt < max_retries - 1:
                sleep_for = _compute_backoff_delay(
                    base_delay=retry_delay,
                    attempt=attempt,
                    backoff_factor=backoff_factor,
                    max_retry_delay=max_retry_delay,
                    jitter_ratio=jitter_ratio,
                )
                time.sleep(sleep_for)
            continue

        return parsed, None

    return None, "; ".join(errors) if errors else "unknown error"
