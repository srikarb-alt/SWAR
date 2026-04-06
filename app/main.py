import os
import time
import logging
from typing import Any, Dict, List

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import pandas as pd
import tempfile

from app.preprocess import preprocess_excel, normalize_cell
from app.processor import process_rows
from app.formatter import client_view, internal_view
from app.llm import call_gemini_with_details, call_gemini_batch_with_details

app = FastAPI()
logger = logging.getLogger("app.main")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default  
    try:
        return int(raw)
    except ValueError:
        return default


JSON_ROW_DELAY_SECONDS = max(0.0, _env_float("JSON_ROW_DELAY_SECONDS", 0.0))
JSON_BATCH_SIZE = max(0, _env_int("JSON_BATCH_SIZE", 0))
JSON_BATCH_PAUSE_SECONDS = max(0.0, _env_float("JSON_BATCH_PAUSE_SECONDS", 0.0))
JSON_MAX_RETRIES = max(1, _env_int("JSON_MAX_RETRIES", 4))
JSON_RETRY_BASE_DELAY_SECONDS = max(0.0, _env_float("JSON_RETRY_BASE_DELAY_SECONDS", 3.0))
JSON_RETRY_BACKOFF_FACTOR = max(1.0, _env_float("JSON_RETRY_BACKOFF_FACTOR", 2.0))
JSON_RETRY_MAX_DELAY_SECONDS = max(0.0, _env_float("JSON_RETRY_MAX_DELAY_SECONDS", 45.0))
JSON_RETRY_JITTER_RATIO = min(1.0, max(0.0, _env_float("JSON_RETRY_JITTER_RATIO", 0.25)))
JSON_LLM_BATCH_SIZE = max(1, _env_int("JSON_LLM_BATCH_SIZE", 1))


class EvaluateJsonItem(BaseModel):
    id: str
    GT: str
    ASR: str = ""


def _ensure_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _build_client_json_row(item_id: str, llm: Dict[str, Any]) -> Dict[str, Any]:
    gt_tokens = _safe_int(llm.get("GT_Tokens", 0))
    exact_words = _ensure_list(llm.get("Exact_Words", []))
    fuzzy_words = _ensure_list(llm.get("Fuzzy_Words", []))
    gt_names = _ensure_list(llm.get("GT_Names", []))
    asr_names = _ensure_list(llm.get("ASR_Names", []))
    gt_numbers = _ensure_list(llm.get("GT_Numbers", []))
    asr_numbers = _ensure_list(llm.get("ASR_Numbers", []))

    exact_count = _safe_int(llm.get("Exact_Count", len(exact_words)))
    fuzzy_count = _safe_int(llm.get("Fuzzy_Count", len(fuzzy_words)))

    matched_count = exact_count + fuzzy_count
    matched_gt_percentage = round((matched_count / gt_tokens) * 100, 2) if gt_tokens else 0.0

    missing_words = _ensure_list(llm.get("Del_Words", []))
    missing_count = len(missing_words)
    missing_percentage = round((missing_count / gt_tokens) * 100, 2) if gt_tokens else 0.0

    gt_names_count = len(gt_names)
    asr_names_count = len(asr_names)
    gt_numbers_count = len(gt_numbers)
    asr_numbers_count = len(asr_numbers)

    return {
        "id": item_id,
        "GT_Translit": llm.get("gt_translit", ""),
        "ASR_Translit": llm.get("asr_translit", ""),
        "Intent_GT": llm.get("Intent_GT", ""),
        "Intent_ASR": llm.get("Intent_ASR", ""),
        "Intent_Similarity_Score": llm.get("Intent_Similarity_Score", 0.0),
        "GT_Tokens": gt_tokens,
        "GT_Names": gt_names,
        "ASR_Names": asr_names,
        "GT_Names_Count": gt_names_count,
        "ASR_Names_Count": asr_names_count,
        "GT_Numbers": gt_numbers,
        "ASR_Numbers": asr_numbers,
        "GT_Numbers_Count": gt_numbers_count,
        "ASR_Numbers_Count": asr_numbers_count,
        "Exact_Words": exact_words,
        "Fuzzy_Words": fuzzy_words,
        "Exact_Count": exact_count,
        "Fuzzy_Count": fuzzy_count,
        "Matched_Count": matched_count,
        "Matched_GT_Percentage": matched_gt_percentage,
        "Missing_Words": missing_words,
        "Missing_Count": missing_count,
        "Missing_Percentage": missing_percentage,
    }


@app.post("/evaluate/client")
async def evaluate_client(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        path = tmp.name

    df = preprocess_excel(path)

    results = process_rows(df)
    out = [client_view(r) for r in results]

    return pd.DataFrame(out).to_dict(orient="records")


@app.post("/evaluate/internal")
async def evaluate_internal(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        path = tmp.name

    df = preprocess_excel(path)

    results = process_rows(df)
    out = [internal_view(r) for r in results]

    return pd.DataFrame(out).to_dict(orient="records")


@app.post("/evaluate/client/json")
async def evaluate_client_json(file: UploadFile = File(...)):
    started_at = time.perf_counter()
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        path = tmp.name

    try:
        df = pd.read_excel(path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to read Excel file ({exc})")

    if not {"Filename", "GT", "ASR"}.issubset(set(df.columns)):
        raise HTTPException(
            status_code=400,
            detail="Excel must contain columns: Filename, GT, ASR",
        )

    results = []
    errors = []
    api_calls_made = 0

    valid_items = []
    for index, row in df.iterrows():
        filename = normalize_cell(row.get("Filename"))
        gt = normalize_cell(row.get("GT"))
        asr = normalize_cell(row.get("ASR"))

        if not filename:
            errors.append({
                "row": index + 1,
                "id": row.get("Filename", ""),
                "error": "Filename is required",
            })
            continue

        if not gt:
            errors.append({
                "row": index + 1,
                "id": filename,
                "error": "GT is required",
            })
            continue

        valid_items.append({
            "row": index + 1,
            "id": filename,
            "gt": gt,
            "asr": asr,
        })

    for start in range(0, len(valid_items), JSON_LLM_BATCH_SIZE):
        batch = valid_items[start:start + JSON_LLM_BATCH_SIZE]
        batch_index = (start // JSON_LLM_BATCH_SIZE) + 1

        if api_calls_made > 0 and JSON_ROW_DELAY_SECONDS > 0:
            time.sleep(JSON_ROW_DELAY_SECONDS)

        logger.info("JSON LLM batch %s (size=%s): starting", batch_index, len(batch))
        llm_batch, err = call_gemini_batch_with_details(
            [{"gt": item["gt"], "asr": item["asr"]} for item in batch],
            max_retries=JSON_MAX_RETRIES,
            retry_delay=JSON_RETRY_BASE_DELAY_SECONDS,
            backoff_factor=JSON_RETRY_BACKOFF_FACTOR,
            max_retry_delay=JSON_RETRY_MAX_DELAY_SECONDS,
            jitter_ratio=JSON_RETRY_JITTER_RATIO,
        )
        api_calls_made += 1

        if JSON_BATCH_SIZE > 0 and api_calls_made % JSON_BATCH_SIZE == 0 and JSON_BATCH_PAUSE_SECONDS > 0:
            time.sleep(JSON_BATCH_PAUSE_SECONDS)

        if llm_batch:
            for item, llm in zip(batch, llm_batch):
                results.append(_build_client_json_row(item_id=item["id"], llm=llm))
            logger.info("JSON LLM batch %s (size=%s): success", batch_index, len(batch))
            continue

        logger.warning(
            "JSON LLM batch %s (size=%s): failed (%s). Falling back to per-row",
            batch_index,
            len(batch),
            err or "unknown error",
        )
        for item in batch:
            llm, per_item_err = call_gemini_with_details(
                item["gt"],
                item["asr"],
                max_retries=JSON_MAX_RETRIES,
                retry_delay=JSON_RETRY_BASE_DELAY_SECONDS,
                backoff_factor=JSON_RETRY_BACKOFF_FACTOR,
                max_retry_delay=JSON_RETRY_MAX_DELAY_SECONDS,
                jitter_ratio=JSON_RETRY_JITTER_RATIO,
            )
            if not llm:
                errors.append({
                    "row": item["row"],
                    "id": item["id"],
                    "error": per_item_err or "Unable to get valid JSON output from LLM",
                })
                logger.warning(
                    "JSON LLM batch %s: per-row failed for id=%s (%s)",
                    batch_index,
                    item["id"],
                    per_item_err or "unknown error",
                )
                continue

            results.append(_build_client_json_row(item_id=item["id"], llm=llm))

    response = {
        "results": results,
        "errors": errors,
        "total": len(valid_items) + len(errors),
        "success_count": len(results),
        "error_count": len(errors),
    }
    elapsed = time.perf_counter() - started_at
    logger.info(
        "JSON request complete: total=%s success=%s errors=%s time=%.2fs",
        response["total"],
        response["success_count"],
        response["error_count"],
        elapsed,
    )
    return response
