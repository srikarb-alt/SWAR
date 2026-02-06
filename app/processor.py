import time
from app.llm import call_gemini

ROW_DELAY = 15


def process_rows(df):
    results = []

    for _, row in df.iterrows():
        llm = call_gemini(row["GT"], row["ASR"])

        if llm:
            llm["Filename"] = row["Filename"]
            results.append(llm)

        time.sleep(ROW_DELAY)

    return results
