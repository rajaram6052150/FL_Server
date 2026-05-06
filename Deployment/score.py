import json
import torch
import numpy as np

from model import ChurnModel


model = None

INPUT_SIZE = 6559


# ==========================================
# LOAD MODEL
# ==========================================

def init():

    global model

    model = ChurnModel(INPUT_SIZE)

    model.load_state_dict(
        torch.load(
            "final_model.pth",
            map_location="cpu"
        )
    )

    model.eval()

    print("Model loaded successfully!")


# ==========================================
# INFERENCE FUNCTION
# ==========================================

def run(raw_data):

    try:

        data = json.loads(raw_data)

        features = np.array(
            data["features"],
            dtype=np.float32
        )

        features = np.expand_dims(
            features,
            axis=0
        )

        x = torch.tensor(features)

        with torch.no_grad():

            output = model(x)

            probability = torch.sigmoid(
                output
            ).item()

        prediction = (
            "Churn"
            if probability > 0.5
            else "No Churn"
        )

        return {
            "prediction": prediction,
            "probability": probability
        }

    except Exception as e:

        return {
            "error": str(e)
        }