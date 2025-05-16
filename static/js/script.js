document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const formData = new FormData(e.target);
    const resultDiv = document.getElementById('result');
    const predictionText = document.getElementById('prediction-text');
    const confidenceText = document.getElementById('confidence-text');

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        if (data.status === 'success') {
            predictionText.textContent = `Prediction: ${data.prediction}`;
            confidenceText.textContent = `Confidence: ${data.confidence}`;
            resultDiv.classList.remove('hidden');
            resultDiv.classList.add('bg-green-100', 'p-4', 'rounded-md');
        } else {
            predictionText.textContent = `Error: ${data.message}`;
            confidenceText.textContent = '';
            resultDiv.classList.remove('hidden');
            resultDiv.classList.add('bg-red-100', 'p-4', 'rounded-md');
        }
    } catch (error) {
        predictionText.textContent = `Error: ${error.message}`;
        confidenceText.textContent = '';
        resultDiv.classList.remove('hidden');
        resultDiv.classList.add('bg-red-100', 'p-4', 'rounded-md');
    }
});