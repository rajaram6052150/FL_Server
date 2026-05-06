"""
Inference API for Federated Learning Churn Model
Loads final global model and exposes prediction endpoint
"""

from flask import Flask, request, jsonify
import torch
import numpy as np

from model import ChurnModel


# ==========================================
# CONFIGURATION
# ==========================================

MODEL_PATH = "models/final_model.pth"

# IMPORTANT:
# Replace with your ACTUAL feature count
INPUT_SIZE = 6559

DEVICE = torch.device("cpu")


# ==========================================
# LOAD MODEL
# ==========================================

model = ChurnModel(INPUT_SIZE).to(DEVICE)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.eval()

print("=" * 60)
print("Federated Learning Inference Server")
print("Model loaded successfully!")
print("=" * 60)


# ==========================================
# CREATE FLASK APP
# ==========================================

app = Flask(__name__)


# ==========================================
# HEALTH CHECK
# ==========================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message": "FL Churn Prediction API Running"
    })


# ==========================================
# PREDICTION ENDPOINT
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.json

        # Expect:
        # {
        #   "features": [ ... ]
        # }

        features = data["features"]

        # Convert to numpy
        features = np.array(
            features,
            dtype=np.float32
        )

        # Shape:
        # [6559] -> [1, 6559]
        features = np.expand_dims(features, axis=0)

        # Convert to tensor
        x = torch.tensor(features).to(DEVICE)

        # Inference
        with torch.no_grad():

            output = model(x)

            probability = torch.sigmoid(output).item()

        prediction = (
            "Churn"
            if probability > 0.5
            else "No Churn"
        )

        return jsonify({

            "prediction": prediction,

            "probability": round(probability, 4)

        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ==========================================
# START SERVER
# ==========================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )