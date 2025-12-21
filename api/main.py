# api/main.py
import os
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional
from api.inference import InferenceService

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)
    topk: Optional[int] = Field(None, ge=1, le=50)
    threshold: Optional[float] = None

class PredictResponse(BaseModel):
    tags: List[str]

app = FastAPI(title="StackOverflow Tagger API", version="1.0.0")

MODEL_DIR = os.getenv("MODEL_DIR", "models")
svc = InferenceService(MODEL_DIR)

@app.on_event("startup")
def startup():
    svc.load()

@app.get("/health")
def health():
    return {"status": "ok", "model_dir": MODEL_DIR}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    tags = svc.predict_tags(req.text, topk=req.topk, threshold=req.threshold)
    return PredictResponse(tags=tags)