# server.py
import flwr as fl
import numpy as np
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics, Parameters, FitRes, EvaluateRes, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy

class FeatureSelectionStrategy(fl.server.strategy.FedAvg):
    """Feature selection strategy using majority voting."""
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, float]]:
        """Aggregate feature masks from clients using majority voting."""
        if not results:
            return None, {}
            
        # Extract feature masks from results
        feature_masks = [parameters_to_ndarrays(fit_res.parameters)[0] for _, fit_res in results]
        
        # Majority voting (features selected by >50% of clients)
        aggregated_mask = np.mean(feature_masks, axis=0) > 0.5
        
        # Convert to int32
        aggregated_mask = aggregated_mask.astype(np.int32)
        
        # Calculate metrics
        selected_count = int(sum(aggregated_mask))
        total_count = len(aggregated_mask)
        
        # Log selected feature count
        print(f"Round {server_round}: Selected {selected_count} features")
        
        # Return parameters AND metrics (this was missing)
        metrics = {"selected_features": selected_count, "total_features": total_count}
        return ndarrays_to_parameters([aggregated_mask]), metrics
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, float]]:
        """Aggregate evaluation results."""
        if not results:
            return None, {}
            
        # Simple averaging of metrics
        coverage_vals = [r.metrics["feature_coverage"] for _, r in results]
        avg_coverage = sum(coverage_vals) / len(coverage_vals)
        
        print(f"Round {server_round}: Average feature coverage: {avg_coverage:.2f}")
        return avg_coverage, {"mean_feature_coverage": avg_coverage}

def main():
    # Configure strategy
    strategy = FeatureSelectionStrategy(
        min_fit_clients=2,
        min_available_clients=2,
        min_evaluate_clients=2,
    )
    
    # Start server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy
    )

if __name__ == "__main__":
    main()