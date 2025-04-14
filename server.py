import flwr as fl
import numpy as np
from typing import Dict, List, Tuple, Optional
from flwr.common import Metrics, Parameters, FitRes, EvaluateRes, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy

class FederatedDPStrategy(fl.server.strategy.FedAvg):
    """Strategy for federated learning with differential privacy."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dp_epsilon = 0.5  # Same epsilon value as clients
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, float]]:
        """Aggregate model parameters, EXCLUDING single-class clients."""
        if not results:
            return None, {}

        # Only use clients with both classes
        filtered = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
            if fit_res.metrics.get("has_both_classes", 0) == 1
        ]
        if not filtered:
            print(f"Round {server_round}: No clients with both classes, skipping aggregation.")
            return None, {}

        total_examples = sum(num_examples for _, num_examples in filtered)
        aggregated_parameters = []
        for i in range(len(filtered[0][0])):
            param_sum = np.sum([params[i] * num_examples / total_examples for params, num_examples in filtered], axis=0)
            aggregated_parameters.append(param_sum)

        # Calculate average metrics
        accuracies = [fit_res.metrics["accuracy"] for _, fit_res in results]
        avg_accuracy = sum(accuracies) / len(accuracies)
        metrics = {
            "accuracy": avg_accuracy,
            "dp_epsilon": self.dp_epsilon,
            "num_clients": len(filtered),
        }
        print(f"Round {server_round}: Aggregated {len(filtered)} clients (excluded single-class clients).")
        return ndarrays_to_parameters(aggregated_parameters), metrics

    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, float]]:
        """Aggregate evaluation results, EXCLUDING single-class clients."""
        if not results:
            return None, {}

        filtered = [
            (eval_res, eval_res.num_examples)
            for _, eval_res in results
            if eval_res.metrics.get("has_both_classes", 0) == 1
        ]
        if not filtered:
            print(f"Round {server_round}: No clients with both classes for evaluation.")
            return None, {}

        total_examples = sum(num_examples for _, num_examples in filtered)
        agg_metrics = {}
        for metric in ["accuracy", "precision", "recall", "f1"]:
            agg_metrics[metric] = sum(
                eval_res.metrics.get(metric, 0) * num_examples / total_examples
                for eval_res, num_examples in filtered
            )
        agg_metrics["dp_epsilon"] = self.dp_epsilon
        agg_metrics["num_clients"] = len(filtered)
        print(f"Round {server_round}: Global model (only both-class clients): "
            f"accuracy={agg_metrics['accuracy']:.4f}, "
            f"precision={agg_metrics['precision']:.4f}, "
            f"recall={agg_metrics['recall']:.4f}, "
            f"f1={agg_metrics['f1']:.4f}")
        return agg_metrics["accuracy"], agg_metrics


def main():
    # Configure strategy
    strategy = FederatedDPStrategy(
        min_fit_clients=2,  
        min_available_clients=2,
        min_evaluate_clients=2
    )
    
    # Start server
    print("Starting server with DP (epsilon=0.5)")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy
    )

if __name__ == "__main__":
    main()