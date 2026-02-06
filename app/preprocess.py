import pandas as pd

def normalize_cell(value):
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    val = str(value).strip()
    if val.lower() in {"nan", "none", "null"}:
        return ""
    return val


def preprocess_excel(file_path):
    df = pd.read_excel(file_path)
    df = df[["Filename", "GT", "ASR"]]

    rows = []
    for _, row in df.iterrows():
        filename = normalize_cell(row["Filename"])
        gt = normalize_cell(row["GT"])
        asr = normalize_cell(row["ASR"])

        if not filename or not gt:
            continue

        rows.append({
            "Filename": filename,
            "GT": gt,
            "ASR": asr
        })

    clean_df = pd.DataFrame(rows).fillna("")
    return clean_df
