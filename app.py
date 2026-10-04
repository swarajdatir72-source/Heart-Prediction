import pickle
import os

# pyright: reportMissingImports=false, reportMissingModuleSource=false
import pandas as pd
import streamlit as st


# ---------------------------------------
# Page configuration
# ---------------------------------------

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)
Heart prediction
# ---------------------------------------
# Paths
# ---------------------------------------

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
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
# UI
# ---------------------------------------

st.title(
    "❤️ Heart Disease Prediction"
)

st.write(
    "Enter patient information below "
    "to generate a machine-learning prediction."
)


# ---------------------------------------
# Input fields
# ---------------------------------------

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=45
)


sex = st.selectbox(
    "Sex",
    ["Male", "Female"]
)

sex = 1 if sex == "Male" else 0


cp = st.selectbox(
    "Chest Pain Type",
    [0, 1, 2, 3]
)


trestbps = st.number_input(
    "Resting Blood Pressure",
    min_value=50,
    max_value=250,
    value=130
)


chol = st.number_input(
    "Cholesterol",
    min_value=50,
    max_value=700,
    value=240
)


fbs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    ["No", "Yes"]
)

fbs = 1 if fbs == "Yes" else 0


restecg = st.selectbox(
    "Resting ECG",
    [0, 1, 2]
)


thalach = st.number_input(
    "Maximum Heart Rate",
    min_value=50,
    max_value=250,
    value=150
)


exang = st.selectbox(
    "Exercise Induced Angina",
    ["No", "Yes"]
)

exang = 1 if exang == "Yes" else 0


oldpeak = st.number_input(
    "Oldpeak",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1
)


slope = st.selectbox(
    "ST Slope",
    [0, 1, 2]
)


ca = st.selectbox(
    "Number of Major Vessels",
    [0, 1, 2, 3]
)


thal = st.selectbox(
    "Thal",
    [0, 1, 2, 3]
)


# ---------------------------------------
# Prediction
# ---------------------------------------

if st.button(
    "🔍 Predict"
):


    input_data = {

        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }


    input_df = pd.DataFrame(
        [input_data]
    )


    input_df = input_df.reindex(
        columns=feature_names,
        fill_value=0
    )


    input_scaled = scaler.transform(
        input_df
    )


    prediction = model.predict(
        input_scaled
    )[0]


    probability = model.predict_proba(
        input_scaled
    )[0]

    confidence = max(
        probability
    ) * 100


    st.divider()


    if prediction == 1:

        st.error(
            "Higher predicted risk"
        )

    else:

        st.success(
            "Lower predicted risk"
        )


    st.metric(
        "Model Confidence",
        f"{confidence:.2f}%"
    )


    st.warning(
        "This is a machine-learning "
        "prediction and is not a medical diagnosis."
    )