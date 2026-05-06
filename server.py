"""
Federated Learning Server
Coordinates training across clients using Flower Framework
Aggregates model updates using FedAvg algorithm
"""

import flwr as fl
from flwr.server.strategy import FedAvg
from typing import List, Tuple

from save_model import save_global_model


# IMPORTANT:
# Replace this with your actual processed feature count
# Example:
# If X_train.shape[1] = 20, then INPUT_SIZE = 20
INPUT_SIZE = 6559


class FedAvgCustom(FedAvg):
    """
    Custom FedAvg strategy with detailed logging
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.round_num = 0

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple],
        failures: List[Tuple],
    ):
        """
        Aggregate model weights using FedAvg
        and save global model after each round
        """

        print(f"\n{'=' * 60}")
        print(f"Federated Round {server_round}")
        print(f"{'=' * 60}")

        print(f"Number of clients completed: {len(results)}")
        print(f"Number of clients failed: {len(failures)}")

        # Print failures if any
        if failures:
            print("\nClient Failures:")
            for client_id, error in failures:
                print(f"Client {client_id} failed: {error}")

        # Perform FedAvg aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round,
            results,
            failures
        )

        # Save aggregated global model
        if aggregated_parameters is not None:

            save_global_model(
                parameters=aggregated_parameters,
                round_num=server_round,
                input_size=INPUT_SIZE
            )

        # Print metrics if available
        if aggregated_metrics:
            print(f"\nAggregated Metrics:")
            print(aggregated_metrics)

        print(f"{'=' * 60}\n")

        return aggregated_parameters, aggregated_metrics


def main():
    """Start Flower Federated Learning Server"""

    print("=" * 60)
    print("Federated Learning Server")
    print("Telco Customer Churn Prediction")
    print("=" * 60)

    print("Server Configuration:")
    print(f"Server Address : 0.0.0.0:8080")
    print(f"Federated Rounds : 5")
    print(f"Minimum Clients : 2")
    print(f"Input Feature Size : {INPUT_SIZE}")

    print("=" * 60)

    # Define Federated Learning Strategy
    strategy = FedAvgCustom(

        # Use all available clients for training
        fraction_fit=1.0,

        # Evaluate on all available clients
        fraction_evaluate=1.0,

        # Minimum clients required for training
        min_fit_clients=2,

        # Minimum clients required for evaluation
        min_evaluate_clients=2,

        # Wait until at least 2 clients connect
        min_available_clients=2,
    )

    # Start Flower Server
    try:

        fl.server.start_server(

            # IMPORTANT:
            # 0.0.0.0 allows external Azure/public access
            server_address="0.0.0.0:8080",

            # Number of federated rounds
            config=fl.server.ServerConfig(
                num_rounds=5
            ),

            strategy=strategy,

            # Prevent message size issues
            grpc_max_message_length=1024 * 1024 * 1024,
        )

    except KeyboardInterrupt:
        print("\n[SERVER STOPPED] Interrupted by user")

    except Exception as e:
        print(f"\n[SERVER ERROR] {e}")

    print("\n" + "=" * 60)
    print("Federated Learning Completed!")
    print("Final global model saved in models/")
    print("=" * 60)


if __name__ == "__main__":
    main()