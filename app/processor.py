import os
import time
import logging
from app.llm import call_gemini, call_gemini_batch_with_details

logger = logging.getLogger("app.processor")

def _env_float(name, default):
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name, default):
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


ROW_DELAY = max(0.0, _env_float("ROW_DELAY_SECONDS", 0.0))
BATCH_SIZE = max(1, _env_int("LLM_BATCH_SIZE", 1))
BATCH_DELAY = max(0.0, _env_float("BATCH_DELAY_SECONDS", 0.0))


def process_rows(df):
    results = []
    records = df.to_dict(orient="records")

    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start:start + BATCH_SIZE]
        batch_index = (start // BATCH_SIZE) + 1

        if BATCH_SIZE == 1:
            row = batch[0]
            logger.info("LLM batch %s (size=1): starting", batch_index)
            llm = call_gemini(row["GT"], row["ASR"])
            if llm:
                llm["Filename"] = row["Filename"]
                results.append(llm)
                logger.info("LLM batch %s (size=1): success", batch_index)
            else:
                logger.warning("LLM batch %s (size=1): failed", batch_index)
            if ROW_DELAY > 0:
                time.sleep(ROW_DELAY)
            continue

        items = [{"gt": row["GT"], "asr": row["ASR"]} for row in batch]
        logger.info("LLM batch %s (size=%s): starting", batch_index, len(items))
        llm_batch, _ = call_gemini_batch_with_details(items)

        if llm_batch:
            for row, llm in zip(batch, llm_batch):
                llm["Filename"] = row["Filename"]
                results.append(llm)
            logger.info("LLM batch %s (size=%s): success", batch_index, len(items))
        else:
            logger.warning(
                "LLM batch %s (size=%s): failed, falling back to per-row",
                batch_index,
                len(items),
            )
            for row in batch:
                llm = call_gemini(row["GT"], row["ASR"])
                if llm:
                    llm["Filename"] = row["Filename"]
                    results.append(llm)
                else:
                    logger.warning("LLM batch %s: per-row call failed", batch_index)
                if ROW_DELAY > 0:
                    time.sleep(ROW_DELAY)

        if BATCH_DELAY > 0:
            time.sleep(BATCH_DELAY)

    return results
