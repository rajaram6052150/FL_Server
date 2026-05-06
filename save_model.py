import os
import torch
from collections import OrderedDict
from flwr.common import parameters_to_ndarrays

from model import ChurnModel


def save_global_model(parameters, round_num, input_size):
    """
    Convert Flower aggregated parameters into a PyTorch model
    and save as .pth checkpoint.

    Args:
        parameters: Aggregated Flower parameters
        round_num: Current federated round
        input_size: Number of model input features
    """

    # Create models directory if not exists
    os.makedirs("models", exist_ok=True)

    # Convert Flower parameters -> NumPy arrays
    ndarrays = parameters_to_ndarrays(parameters)

    # Initialize model
    model = ChurnModel(input_size=input_size)

    # Map weights to model state_dict
    params_dict = zip(model.state_dict().keys(), ndarrays)

    state_dict = OrderedDict({
        k: torch.tensor(v)
        for k, v in params_dict
    })

    # Load weights into model
    model.load_state_dict(state_dict, strict=True)

    # Save round checkpoint
    round_path = f"models/global_model_round_{round_num}.pth"

    torch.save(model.state_dict(), round_path)

    print(f"[MODEL SAVED] {round_path}")

    # Optional: overwrite latest model
    latest_path = "models/final_model.pth"

    torch.save(model.state_dict(), latest_path)

    print(f"[LATEST MODEL UPDATED] {latest_path}")


