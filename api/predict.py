import os
import pickle

try:
    import pandas as pd  # type: ignore[import-not-found]
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "pandas is required to run the prediction API. "
        "Install the project dependencies first."
    ) from exc

try:
    from fastapi import FastAPI  # type: ignore[import-not-found]
    from fastapi.middleware.cors import CORSMiddleware  # type: ignore[import-not-found]
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "fastapi is required to run the prediction API. "
        "Install the project dependencies first."
    ) from exc

try:
    from pydantic import BaseModel  # type: ignore[import-not-found]
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "pydantic is required to run the prediction API. "
        "Install the project dependencies first."
    ) from exc


# ---------------------------------------
# Paths
# ---------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "heart_model.pkl"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "model",
    "scaler.pkl"
)

FEATURES_PATH = os.path.join(
    BASE_DIR,
    "model",
    "features.pkl"
)


# ---------------------------------------
# Load model
# ---------------------------------------

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


with open(SCALER_PATH, "rb") as file:
    scaler = pickle.load(file)


with open(FEATURES_PATH, "rb") as file:
    feature_names = pickle.load(file)


# ---------------------------------------
# FastAPI app
# ---------------------------------------

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Machine Learning API for heart disease prediction",
    version="1.0.0"
)


# ---------------------------------------
# CORS
# ---------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------
# Input data
# ---------------------------------------

class HeartData(BaseModel):

    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float


# ---------------------------------------
# Home endpoint
# ---------------------------------------

@app.get("/")
def home():

    return {
        "message": "Heart Disease Prediction API",
        "status": "running"
    }


# ---------------------------------------
# Health check
# ---------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ---------------------------------------
# Prediction endpoint
# ---------------------------------------

@app.post("/predict")
def predict(data: HeartData):

    # Convert input to dictionary
    input_data = {
        "age": data.age,
        "sex": data.sex,
        "cp": data.cp,
        "trestbps": data.trestbps,
        "chol": data.chol,
        "fbs": data.fbs,
        "restecg": data.restecg,
        "thalach": data.thalach,
        "exang": data.exang,
        "oldpeak": data.oldpeak,
        "slope": data.slope,
        "ca": data.ca,
        "thal": data.thal
    }


    # Create DataFrame
    input_df = pd.DataFrame(
        [input_data]
    )


    # Convert categorical values
    input_df = pd.get_dummies(
        input_df,
        drop_first=False
    )


    # Match training features
    input_df = input_df.reindex(
        columns=feature_names,
        fill_value=0
    )


    # Scale input
    input_scaled = scaler.transform(
        input_df
    )


    # Prediction
    prediction = model.predict(
        input_scaled
    )[0]


    # Probability
    probabilities = model.predict_proba(
        input_scaled
    )[0]

    probability = float(
        max(probabilities)
    )


    # Result
    if prediction == 1:

        result = "Higher predicted risk"

    else:

        result = "Lower predicted risk"


    return {
        "prediction": int(prediction),
        "result": result,
        "probability": round(
            probability * 100,
            2
        ),
        "message": (
            "This result is a machine-learning "
            "prediction and is not a medical diagnosis."
        )
    }