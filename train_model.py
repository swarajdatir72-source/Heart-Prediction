
from pathlib import Path
import importlib
import pickle
import pandas as pd

try:
    train_test_split = importlib.import_module("sklearn.model_selection").train_test_split
    RandomForestClassifier = importlib.import_module("sklearn.ensemble").RandomForestClassifier
    SimpleImputer = importlib.import_module("sklearn.impute").SimpleImputer
    Pipeline = importlib.import_module("sklearn.pipeline").Pipeline
    metrics = importlib.import_module("sklearn.metrics")
    accuracy_score = metrics.accuracy_score
    classification_report = metrics.classification_report
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "scikit-learn is required to train the model. "
        "Install it with: pip install scikit-learn"
    ) from exc

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = Path(
    r"C:\Users\USER\heart-disease-prediction\data\heart.csv"
)

MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "heart_model.pkl"
FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal"
]

TARGET_CANDIDATES = ["target", "num", "HeartDisease", "heart_disease"]

if not DATA_PATH.exists():
    raise FileNotFoundError("Place heart.csv in the project root.")

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

# Handle common target-column naming variations.
target_col = next(
    (c for c in TARGET_CANDIDATES if c in df.columns),
    None
)

if target_col is None:
    raise ValueError(
        f"Target column not found. Dataset columns: {df.columns.tolist()}"
    )

missing = [c for c in FEATURES if c not in df.columns]
if missing:
    raise ValueError(
        f"Missing expected features: {missing}. "
        "Update FEATURES to match your dataset."
    )

# Convert fields to numeric; unparseable values become missing.
X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(df[target_col], errors="coerce")

valid = y.notna()
X = X.loc[valid]
y = y.loc[valid]

# Most common heart.csv datasets use 0 = absent and
# positive values = disease. Confirm this for your dataset.
y = (y > 0).astype(int)

if y.nunique() != 2:
    raise ValueError("The target must contain both classes: 0 and 1.")

if y.value_counts().min() < 2:
    raise ValueError("Each target class needs at least two rows.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)

print("Test accuracy:", round(accuracy_score(y_test, predictions), 4))
print(classification_report(y_test, predictions))

MODEL_DIR.mkdir(exist_ok=True)

with open(MODEL_PATH, "wb") as f:
    pickle.dump({
        "model": pipeline,
        "features": FEATURES
    }, f)

print(f"Model saved to: {MODEL_PATH}")
