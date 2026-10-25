
from pathlib import Path
import pickle
import math

from fastapi import FastAPI, HTTPException  # pyright: ignore[reportMissingImports]
from fastapi.middleware.cors import CORSMiddleware  # pyright: ignore[reportMissingImports]
from pydantic import BaseModel, Field # pyright: ignore[reportMissingImports]

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "heart_model.pkl"

app = FastAPI(title="Heart Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

try:
    with open(MODEL_PATH, "rb") as f:
        saved = pickle.load(f)
    model = saved["model"]
    FEATURES = saved["features"]
except Exception as exc:
    model = None
    FEATURES = []
    print(f"Model loading error: {exc}")


class PatientData(BaseModel):
    age: int = Field(ge=1, le=120)
    sex: int = Field(ge=0, le=1)
    cp: int = Field(ge=0, le=3)
    trestbps: int = Field(ge=40, le=300)
    chol: int = Field(ge=50, le=800)
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    thalach: int = Field(ge=40, le=250)
    exang: int = Field(ge=0, le=1)
    oldpeak: float = Field(ge=0, le=15)
    slope: int = Field(ge=0, le=2)
    ca: int = Field(ge=0, le=4)
    thal: int = Field(ge=0, le=3)


@app.get("/api/health")
def health():
    return {
        "status": "running",
        "model_loaded": model is not None
    }


@app.post("/api/predict")
def predict(patient: PatientData):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model unavailable. Train the model first."
        )

    values = patient.model_dump()
    row = [[values[name] for name in FEATURES]]

    try:
        prediction = int(model.predict(row)[0])
        probabilities = model.predict_proba(row)[0]
        confidence = float(max(probabilities)) * 100

        if not math.isfinite(confidence):
            raise ValueError("Invalid probability")

        return {
            "prediction": prediction,
            "label": (
                "Higher predicted risk"
                if prediction == 1
                else "Lower predicted risk"
            ),
            "confidence": round(confidence, 1),
            "notice": (
                "Educational model output only; "
                "not a medical diagnosis."
            )
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check model compatibility."
        ) from exc
from fastapi.staticfiles import StaticFiles  # pyright: ignore[reportMissingImports]

app.mount(
    "/",
    StaticFiles(directory=str(BASE_DIR / "public"), html=True),
    name="public"
)