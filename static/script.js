function analyzeSentiment() {
  const review = document.getElementById("review").value;

  fetch("/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ review: review })
  })
  .then(response => response.json())
  .then(data => {
    const resultDiv = document.getElementById("result");
    if (data.sentiment) {
      resultDiv.textContent = "Sentiment: " + data.sentiment.toUpperCase();
    } else if (data.error) {
      resultDiv.textContent = "Error: " + data.error;
    }
  })
  .catch(error => {
    console.error("Error:", error);
  });
}
