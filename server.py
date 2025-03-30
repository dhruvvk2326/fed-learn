import flwr as fl
import numpy as np

# Simple weighted aggregation function (using a dummy weighted average for demonstration)
def weighted_average(metrics):
    accuracies = [num_examples * metric["accuracy"] for num_examples, metric in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return 0
    return sum(accuracies) / total_examples

# Start the federated server with FedAvg strategy
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average)
)
