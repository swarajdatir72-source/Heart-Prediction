
const form = document.getElementById("predictionForm");
const button = document.getElementById("predictBtn");

const resultTitle = document.getElementById("resultTitle");
const resultText = document.getElementById("resultText");
const resultClass = document.getElementById("resultClass");
const resultScore = document.getElementById("resultScore");
const errorMessage = document.getElementById("errorMessage");
const notice = document.getElementById("notice");

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  errorMessage.hidden = true;
  errorMessage.textContent = "";

  const formData = new FormData(form);
  const data = {};

  for (const [key, value] of formData.entries()) {
    data[key] = Number(value);
  }

  if (!form.reportValidity()) return;

  button.disabled = true;
  button.querySelector("span:first-child").textContent = "Analyzing...";
  resultTitle.textContent = "Processing";
  resultText.textContent = "The model is evaluating the submitted parameters.";
  resultClass.textContent = "—";
  resultScore.textContent = "—";

  try {
    // Relative URL works locally and on the same Vercel deployment.
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data)
    });

    const result = await response.json();

    if (!response.ok) {
      throw new Error(
        typeof result.detail === "string"
          ? result.detail
          : "The prediction request failed."
      );
    }

    const higherRisk = Number(result.prediction) === 1;

    resultTitle.textContent = higherRisk
      ? "Higher predicted risk"
      : "Lower predicted risk";

    resultText.textContent = higherRisk
      ? "The model assigned the higher-risk class. This is not a diagnosis."
      : "The model assigned the lower-risk class. This does not rule out disease.";

    resultClass.textContent = higherRisk ? "Class 1" : "Class 0";

    resultScore.textContent =
      Number.isFinite(Number(result.confidence))
        ? `${Number(result.confidence).toFixed(1)}%`
        : "Unavailable";

    notice.textContent = result.notice ||
      "Educational output only. Consult a healthcare professional.";

  } catch (error) {
    resultTitle.textContent = "Unable to predict";
    resultText.textContent = "Check the backend and try again.";
    resultClass.textContent = "—";
    resultScore.textContent = "—";
    errorMessage.textContent = error.message ||
      "A network error occurred.";
    errorMessage.hidden = false;
  } finally {
    button.disabled = false;
    button.querySelector("span:first-child").textContent =
      "Analyze parameters";
  }
});
