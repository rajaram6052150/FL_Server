"""
Federated Learning Server
Telco Customer Churn Prediction
Flower + PyTorch
"""

import os
import flwr as fl
import torch

from flwr.server.strategy import FedAvg
from flwr.common import parameters_to_ndarrays
from typing import List, Tuple

from model import ChurnModel


# =========================================================
# CREATE MODEL DIRECTORY
# =========================================================

os.makedirs("models", exist_ok=True)


# =========================================================
# CUSTOM FEDAVG STRATEGY
# =========================================================

class FedAvgCustom(FedAvg):

    def __init__(self, input_size, num_rounds, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.input_size = input_size

        self.num_rounds = num_rounds

    # =====================================================
    # AGGREGATE CLIENT WEIGHTS
    # =====================================================

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple],
        failures: List[Tuple],
    ):

        print("\n" + "=" * 60)
        print(f"Federated Round {server_round}")
        print("=" * 60)

        print(f"Clients completed : {len(results)}")
        print(f"Clients failed    : {len(failures)}")

        # =================================================
        # PRINT FAILURES
        # =================================================

        if failures:

            print("\nFailed Clients:")

            for client_id, error in failures:

                print(f"{client_id} -> {error}")

        # =================================================
        # FEDAVG AGGREGATION
        # =================================================

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round,
            results,
            failures,
        )

        # =================================================
        # SAVE GLOBAL MODEL
        # =================================================

        if aggregated_parameters is not None:

            print("\nSaving global model...")

            # Convert Flower Parameters -> NumPy Arrays
            aggregated_ndarrays = parameters_to_ndarrays(
                aggregated_parameters
            )

            # Create model architecture
            model = ChurnModel(self.input_size)

            # Get current state_dict
            state_dict = model.state_dict()

            # Match keys with parameters
            params_dict = zip(
                state_dict.keys(),
                aggregated_ndarrays
            )

            # Convert NumPy -> Torch Tensor
            state_dict = {
                k: torch.tensor(v)
                for k, v in params_dict
            }

            # Load aggregated weights
            model.load_state_dict(
                state_dict,
                strict=True
            )

            # =================================================
            # SAVE ROUND CHECKPOINT
            # =================================================

            round_model_path = (
                f"models/global_model_round_{server_round}.pth"
            )

            torch.save(
                model.state_dict(),
                round_model_path
            )

            print(f"Round model saved: {round_model_path}")

            # =================================================
            # SAVE FINAL MODEL
            # =================================================

            if server_round == self.num_rounds:

                final_model_path = (
                    "models/final_model.pth"
                )

                torch.save(
                    model.state_dict(),
                    final_model_path
                )

                print("\n" + "=" * 60)
                print("FINAL GLOBAL MODEL SAVED!")
                print(f"Location: {final_model_path}")
                print("=" * 60)

        return aggregated_parameters, aggregated_metrics


# =========================================================
# MAIN FUNCTION
# =========================================================

def main():

    # =====================================================
    # CONFIG
    # =====================================================

    INPUT_SIZE = 6559

    NUM_ROUNDS = 5

    # =====================================================
    # SERVER INFO
    # =====================================================

    print("=" * 60)
    print("Federated Learning Server")
    print("Telco Customer Churn Prediction")
    print("=" * 60)

    print("Server Configuration:")

    print(f"Server Address   : 0.0.0.0:8080")

    print(f"Federated Rounds : {NUM_ROUNDS}")

    print(f"Minimum Clients  : 2")

    print(f"Input Size       : {INPUT_SIZE}")

    print("=" * 60)

    # =====================================================
    # FEDAVG STRATEGY
    # =====================================================

    strategy = FedAvgCustom(

        input_size=INPUT_SIZE,

        num_rounds=NUM_ROUNDS,

        fraction_fit=1.0,

        fraction_evaluate=1.0,

        min_fit_clients=2,

        min_evaluate_clients=2,

        min_available_clients=2,
    )

    # =====================================================
    # START FLOWER SERVER
    # =====================================================

    try:

        fl.server.start_server(

            server_address="0.0.0.0:8080",

            config=fl.server.ServerConfig(
                num_rounds=NUM_ROUNDS
            ),

            strategy=strategy,
        )

    except KeyboardInterrupt:

        print("\nServer interrupted by user")

    except Exception as e:

        print(f"\n[SERVER ERROR] {e}")

    print("\n" + "=" * 60)

    print("Federated Learning Completed!")

    print("Final model available in models/")

    print("=" * 60)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()