from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import tempfile

from app.preprocess import preprocess_excel
from app.processor import process_rows
from app.formatter import client_view, internal_view

app = FastAPI()


@app.post("/evaluate/client")
async def evaluate_client(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        path = tmp.name

    df = preprocess_excel(path)

    if len(df) > 20:
        raise HTTPException(400, "Max 20 rows allowed")

    results = process_rows(df)
    out = [client_view(r) for r in results]

    return pd.DataFrame(out).to_dict(orient="records")


@app.post("/evaluate/internal")
async def evaluate_internal(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        path = tmp.name

    df = preprocess_excel(path)

    if len(df) > 20:
        raise HTTPException(400, "Max 20 rows allowed")

    results = process_rows(df)
    out = [internal_view(r) for r in results]

    return pd.DataFrame(out).to_dict(orient="records")
