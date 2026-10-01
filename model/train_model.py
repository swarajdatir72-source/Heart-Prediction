import os
import pickle
import pandas as pd  # type: ignore[import-not-found]

from sklearn.model_selection import train_test_split  # type: ignore[import-not-found]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-not-found]
from sklearn.ensemble import RandomForestClassifier  # type: ignore[import-not-found]
from sklearn.metrics import (  # type: ignore[import-not-found]
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ---------------------------------------
# 1. Paths
# ---------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "heart.csv"
)

MODEL_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "heart_model.pkl"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "scaler.pkl"
)

FEATURES_PATH = os.path.join(
    MODEL_DIR,
    "features.pkl"
)


# ---------------------------------------
# 2. Load dataset
# ---------------------------------------

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


# ---------------------------------------
# 3. Clean column names
# ---------------------------------------

df.columns = df.columns.str.strip().str.lower()

print("\nCleaned columns:")
print(df.columns.tolist())


# ---------------------------------------
# 4. Find target column
# ---------------------------------------

possible_targets = [
    "target",
    "condition",
    "output",
    "heartdisease",
    "heart_disease",
    "num"
]

target_column = None

for column in possible_targets:
    if column in df.columns:
        target_column = column
        break

if target_column is None:
    raise ValueError(
        "Target column not found. "
        "Rename your target column to 'target'."
    )

print("\nTarget column:", target_column)


# ---------------------------------------
# 5. Remove missing values
# ---------------------------------------

print("\nMissing values:")
print(df.isnull().sum())

df = df.dropna()

print("\nDataset after removing missing values:")
print(df.shape)


# ---------------------------------------
# 6. Separate features and target
# ---------------------------------------

X = df.drop(columns=[target_column])
y = df[target_column]


# ---------------------------------------
# 7. Convert categorical/object columns
# ---------------------------------------

X = pd.get_dummies(
    X,
    drop_first=False
)


# Save exact feature names
feature_names = X.columns.tolist()


# ---------------------------------------
# 8. Train-test split
# ---------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ---------------------------------------
# 9. Feature scaling
# ---------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ---------------------------------------
# 10. Create Random Forest model
# ---------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1
)


# ---------------------------------------
# 11. Train model
# ---------------------------------------

print("\nTraining model...")

model.fit(
    X_train_scaled,
    y_train
)


# ---------------------------------------
# 12. Prediction
# ---------------------------------------

y_pred = model.predict(
    X_test_scaled
)


# ---------------------------------------
# 13. Evaluation
# ---------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n===================================")
print("MODEL PERFORMANCE")
print("===================================")

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ---------------------------------------
# 14. Save model
# ---------------------------------------

with open(
    MODEL_PATH,
    "wb"
) as file:

    pickle.dump(
        model,
        file
    )


# ---------------------------------------
# 15. Save scaler
# ---------------------------------------

with open(
    SCALER_PATH,
    "wb"
) as file:

    pickle.dump(
        scaler,
        file
    )


# ---------------------------------------
# 16. Save feature names
# ---------------------------------------

with open(
    FEATURES_PATH,
    "wb"
) as file:

    pickle.dump(
        feature_names,
        file
    )


print("\n===================================")
print("MODEL SAVED SUCCESSFULLY")
print("===================================")

print("\nFiles created:")

print(MODEL_PATH)
print(SCALER_PATH)
print(FEATURES_PATH)