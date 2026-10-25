# ❤️ Heart Disease Prediction

A machine learning based web application for heart disease risk prediction.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest
- FastAPI
- HTML
- CSS
- JavaScript
- Vercel

## Project Structure

heart-disease-prediction/

├── data/
│   └── heart.csv
│
├── model/
│   ├── train_model.py
│   ├── heart_model.pkl
│   ├── scaler.pkl
│   └── features.pkl
│
├── api/
│   └── predict.py
│
├── public/
│   └── index.html
│
├── app.py
├── requirements.txt
├── vercel.json
├── .gitignore
└── README.md

## How to Run

### 1. Create virtual environment

python -m venv venv

### 2. Activate environment

Windows:

venv\Scripts\activate

### 3. Install dependencies

pip install -r requirements.txt

### 4. Train model

python model/train_model.py

### 5. Run FastAPI locally

uvicorn api.predict:app --reload

### 6. Open API

http://127.0.0.1:8000

### 7. Run Streamlit version

streamlit run app.py

## Deployment

Push the project to GitHub and deploy the repository using Vercel.

## Disclaimer

This application provides a machine-learning prediction and is not a medical diagnosis.