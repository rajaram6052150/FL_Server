"""
Federated Learning Server
Coordinates training across clients using Flower Framework
Aggregates model updates using FedAvg algorithm
"""

import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Metrics
from typing import Dict, List, Tuple, Optional, Union
import time


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
        Aggregate model weights using simple averaging (FedAvg)
        """
        print(f"\n=== Federated Round {server_round} ===")
        print(f"Number of clients completed: {len(results)}")
        print(f"Number of clients failed: {len(failures)}")
        
        if failures:
            for client_id, error in failures:
                print(f"Client {client_id} failed: {error}")
        
        # Aggregate
        aggregated_weights, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        if aggregated_metrics:
            print(f"Aggregated metrics: {aggregated_metrics}")
        
        return aggregated_weights, aggregated_metrics


def main():
    """Start Flower Server"""
    
    print("=" * 60)
    print("Federated Learning Server - Telco Customer Churn Prediction")
    print("=" * 60)
    print(f"Server starting on localhost:8080")
    print(f"Number of rounds: 5")
    print(f"Min clients: 2")
    print(f"Min available: 2")
    print("=" * 60)
    
    # Define strategy
    strategy = FedAvgCustom(
        fraction_fit=1.0,  # Use all available clients
        fraction_evaluate=1.0,  # Evaluate on all clients
        min_fit_clients=2,  # Minimum 2 clients to proceed
        min_evaluate_clients=2,
        min_available_clients=2,  # Wait for at least 2 clients to be available
    )
    
    # Start server
    try:
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=5),
            strategy=strategy,
        )
    except KeyboardInterrupt:
        print("\nServer interrupted by user")
    except Exception as e:
        print(f"\nServer error: {e}")
    
    print("\n" + "=" * 60)
    print("Federated Learning Completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
