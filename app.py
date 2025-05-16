from flask import Flask, render_template, request, jsonify
import joblib
from src.constants import MODEL_DIR_PATH, PREPROCESSOR_DIR_PATH, ENCODER_DIR_PATH
import numpy as np
from src.logger.custom_logger import logger

app = Flask(__name__)

# Load the pre-trained model
try:
    model = joblib.load(MODEL_DIR_PATH)
except FileNotFoundError:
    print("Model file not found. Please ensure model .pkl file exists.")
    exit(1)
try:
    scaler = joblib.load(PREPROCESSOR_DIR_PATH)
    encoder = joblib.load(ENCODER_DIR_PATH)
except Exception as e:
    logger.error(e)


# Define feature columns (excluding 'class')
feature_columns = [
    "age",
    "gender",
    "polyuria",
    "polydipsia",
    "sudden_weight_loss",
    "weakness",
    "polyphagia",
    "genital_thrush",
    "visual_blurring",
    "itching",
    "irritability",
    "delayed_healing",
    "partial_paresis",
    "muscle_stiffness",
    "alopecia",
    "obesity",
]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract form data
        data = request.form
        features = []

        # Process numerical feature: age
        age = int(data.get("age", 0))

        features.append(age)

        # Process binary/categorical features
        gender = data.get("gender")
        gender = encoder.named_steps["One Hot Encoder"].transform([[gender]])
        gender = gender.flatten()  # Flatten the one-hot encoded array
        logger.info(gender)
        features.extend(gender)  # Use extend instead of append to add individual elements

        # Process yes/no features
        for feature in feature_columns[2:]:
            value = 1 if data.get(feature, "No") == "Yes" else 0
            features.append(value)

        # Convert to numpy array for prediction
        features_array = np.array([features])

        # logger.info(f'{encoder.named_steps["One Hot Encoder"]},{model},{scaler}')
        logger.info(f"User has Entered these Details {features_array}")

        # Make prediction
        prediction = model.named_steps["rf"].predict(features_array)
        probability = model.predict_proba(features_array)[0][1]

        # Prepare response
        result = "Positive" if prediction == 1 else "Negative"
        confidence = f"{probability * 100:.2f}%"

        return jsonify({"status": "success", "prediction": result, "confidence": confidence})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
