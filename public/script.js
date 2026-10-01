const form = document.getElementById('predict-form');
const resultBox = document.getElementById('result-box');
const riskTag = document.getElementById('risk-tag');
const probabilityValue = document.getElementById('probability-value');
const probabilityBar = document.getElementById('probability-bar');
const resultMessage = document.getElementById('result-message');

function getApiUrl() {
  const origin = window.location.origin;
  const isLocalFile = window.location.protocol === 'file:';

  if (isLocalFile) {
    return 'http://127.0.0.1:8000/predict';
  }

  return `${origin}/api/predict`;
}

function buildPayload() {
  const fields = [
    'age',
    'sex',
    'cp',
    'trestbps',
    'chol',
    'fbs',
    'restecg',
    'thalach',
    'exang',
    'oldpeak',
    'slope',
    'ca',
    'thal'
  ];

  const payload = {};

  fields.forEach((field) => {
    const value = document.getElementById(field).value;
    payload[field] = Number(value);
  });

  return payload;
}

function showResult(prediction, probability, text) {
  const isHighRisk = prediction === 1;
  riskTag.textContent = isHighRisk ? 'High Risk' : 'Low Risk';
  riskTag.classList.toggle('high-risk', isHighRisk);
  probabilityValue.textContent = `${Math.round(probability)}%`;
  probabilityBar.style.width = `${Math.min(probability, 100)}%`;
  resultMessage.textContent = text;
  resultBox.classList.remove('hidden');
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();

  const payload = buildPayload();

  try {
    const response = await fetch(getApiUrl(), {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      throw new Error(`Request failed with status ${response.status}`);
    }

    const data = await response.json();
    const probability = Number(data.probability || 0);
    const prediction = Number(data.prediction || 0);

    const riskText = prediction === 1
      ? 'This patient may have a higher risk of heart disease based on the entered indicators.'
      : 'This patient appears to have a lower risk profile based on the entered indicators.';

    showResult(prediction, probability, `${riskText} ${data.message || ''}`);
  } catch (error) {
    showResult(0, 0, 'Unable to connect to the prediction API. Please ensure the backend is running and try again.');
    console.error(error);
  }
});
