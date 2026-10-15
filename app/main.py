from pathlib import Path
import json
import joblib
import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path("/content/coughvid_project")
MODEL_DIR = BASE_DIR / "tuned_models"

app = FastAPI(
    title="CoughVID Research API",
    description="Research classification of cough audio features",
    version="1.0.0"
)

MODEL_FILES = {
    "XGBoost": "xgboost.joblib",
    "Random Forest": "random_forest.joblib",
    "Linear SVM": "linear_svm.joblib",
}

models = {}
for name, filename in MODEL_FILES.items():
    path = MODEL_DIR / filename
    if path.exists():
        models[name] = joblib.load(path)

metrics_path = MODEL_DIR / "comparison.json"
metrics = (
    json.loads(metrics_path.read_text())
    if metrics_path.exists()
    else []
)


class PredictionRequest(BaseModel):
    # Temporary endpoint: accepts an already-extracted
    # feature vector, not raw audio.
    features: list[float] = Field(min_length=1)
    model_name: str = "XGBoost"


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "loaded_models": list(models.keys())
    }


@app.get("/metrics")
def get_metrics():
    if not metrics:
        raise HTTPException(
            status_code=404,
            detail="Comparison metrics not found"
        )
    return {"model_comparison": metrics}


@app.post("/predict")
def predict(request: PredictionRequest):
    model = models.get(request.model_name)

    if model is None:
        raise HTTPException(
            status_code=400,
            detail=f"Model unavailable. Choose one of: {list(models)}"
        )

    features = np.asarray(
        request.features, dtype=np.float32
    ).reshape(1, -1)

    expected = getattr(model, "n_features_in_", None)
    if expected is not None and features.shape[1] != expected:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {expected} features, "
                   f"received {features.shape[1]}"
        )

    # Reuse the validation-selected threshold.
    row = next(
        (r for r in metrics
         if r["model"] == request.model_name),
        None
    )
    if row is None:
        raise HTTPException(
            status_code=500,
            detail="Model threshold is unavailable"
        )

    if request.model_name == "Linear SVM":
        score = float(model.decision_function(features)[0])
    else:
        score = float(model.predict_proba(features)[0, 1])

    threshold = float(row["threshold"])
    prediction = int(score >= threshold)

    return {
        "model": request.model_name,
        "prediction": prediction,
        "label": (
            "COVID-19-labelled"
            if prediction == 1
            else "healthy-labelled"
        ),
        "score": score,
        "threshold": threshold,
        "notice": (
            "Research output only. This is not a medical diagnosis."
        )
    }
