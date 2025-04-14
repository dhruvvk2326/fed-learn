import os
import pandas as pd
import numpy as np
import flwr as fl
import argparse
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.dummy import DummyClassifier

class NetworkIntrusionClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        client_file = f"client_{client_id}.parquet"
        if not os.path.exists(client_file):
            raise FileNotFoundError(f"Client data not found: {client_file}")

        self.df = pd.read_parquet(client_file)
        print(f"Client {client_id} original classes: {sorted(self.df['Label'].unique())}")
        self.df['is_attack'] = np.where(self.df['Label'] == 'Benign', 0, 1)

        attack_count = self.df['is_attack'].sum()
        benign_count = len(self.df) - attack_count
        print(f"Client {client_id}: {benign_count} benign, {attack_count} attack samples")

        self.has_both_classes = attack_count > 0 and benign_count > 0
        if self.has_both_classes:
            class_weights = {
                0: 1.0,
                1: min(benign_count / max(1, attack_count), 10.0)
            }
            print(f"Client {client_id} using all {len(self.df)} samples with class weights: {class_weights}")

        self.X = self.df.drop(columns=['Label', 'is_attack']).values
        self.y = self.df['is_attack'].values

        try:
            if os.path.exists("robust_scaler_new.pkl"):
                self.scaler = joblib.load("robust_scaler_new.pkl")
                self.X = self.scaler.transform(self.X)
            elif os.path.exists("robust_scaler.pkl"):
                self.scaler = joblib.load("robust_scaler.pkl")
                self.X = self.scaler.transform(self.X)
        except Exception as e:
            print(f"Client {client_id}: Warning: Could not load/apply scaler: {e}")

        if self.has_both_classes:
            self.model = LogisticRegression(
                penalty='l2', C=0.1, solver='liblinear', max_iter=1000,
                class_weight=class_weights, warm_start=True
            )
        else:
            print(f"Client {client_id} has only one class: {np.unique(self.y)[0]}")
            self.model = DummyClassifier(strategy="constant", constant=np.unique(self.y)[0])

        self.base_epsilon = 0.5
        self.sensitivity = 2.0 if (self.has_both_classes and attack_count > 100) else 0.5
        print(f"Client {client_id} initialized with {len(self.X)} samples, sensitivity={self.sensitivity}")

    def add_noise_to_parameters(self, parameters, round_num):
        delta = 1e-5
        epsilon = self.base_epsilon * (1 + 0.1 * round_num)  # Gradual privacy relaxation
        noise_scale = np.sqrt(2 * np.log(1.25/delta)) * self.sensitivity / epsilon

        noisy_parameters = []
        for param in parameters:
            param_clipped = np.clip(param, -self.sensitivity, self.sensitivity)
            if param.ndim > 1 and self.has_both_classes:
                coef_magnitude = np.abs(param_clipped)
                adaptive_scale = 1.0 / (1.0 + coef_magnitude)
                scaled_noise = np.random.normal(0, noise_scale, param.shape) * adaptive_scale
                noisy_parameters.append(param_clipped + scaled_noise)
            else:
                noise = np.random.normal(0, noise_scale, param.shape)
                noisy_parameters.append(param_clipped + noise)

        return noisy_parameters

    def get_parameters(self, config):
        round_num = config.get("server_round", 1)
        if isinstance(self.model, LogisticRegression) and hasattr(self.model, 'coef_'):
            parameters = [
                self.model.coef_.astype(np.float32),
                self.model.intercept_.astype(np.float32)
            ]
        else:
            n_features = self.X.shape[1]
            parameters = [
                np.zeros((1, n_features), dtype=np.float32),
                np.zeros(1, dtype=np.float32)
            ]

        return self.add_noise_to_parameters(parameters, round_num)

    def set_parameters(self, parameters):
        if isinstance(self.model, LogisticRegression):
            n_features = self.X.shape[1]
            coef = parameters[0].reshape(1, n_features)
            intercept = parameters[1].reshape(1)
            if not hasattr(self.model, 'classes_'):
                self.model.fit(self.X[:10], np.array([0, 1]*5))
            self.model.coef_ = coef
            self.model.intercept_ = intercept
            self.model.classes_ = np.array([0, 1])
        elif isinstance(self.model, DummyClassifier) and not hasattr(self.model, 'classes_'):
            self.model.fit(self.X[:1], self.y[:1])

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        try:
            self.model.fit(self.X, self.y)
        except Exception as e:
            print(f"Client {self.client_id} - Fit error: {str(e)}")

        params = self.get_parameters(config)
        y_pred = self.model.predict(self.X)
        accuracy = accuracy_score(self.y, y_pred)

        metrics = {"accuracy": float(accuracy), "has_both_classes": int(self.has_both_classes)}
        if self.has_both_classes:
            metrics.update({
                "precision": float(precision_score(self.y, y_pred, zero_division=0)),
                "recall": float(recall_score(self.y, y_pred, zero_division=0)),
                "f1": float(f1_score(self.y, y_pred, zero_division=0))
            })

        print(f"Client {self.client_id} - Training metrics: accuracy={accuracy:.4f}, "
              f"DP epsilon={self.base_epsilon * (1 + 0.1 * config.get('server_round', 1)):.2f}")
        return params, len(self.X), metrics

    def evaluate(self, parameters, config):
        try:
            self.set_parameters(parameters)
            y_pred = self.model.predict(self.X)
            accuracy = accuracy_score(self.y, y_pred)
            metrics = {"accuracy": float(accuracy)}
            if self.has_both_classes and len(np.unique(y_pred)) > 1:
                metrics.update({
                    "precision": float(precision_score(self.y, y_pred, zero_division=0)),
                    "recall": float(recall_score(self.y, y_pred, zero_division=0)),
                    "f1": float(f1_score(self.y, y_pred, zero_division=0)),
                    "has_both_classes": 1
                })
            else:
                metrics["has_both_classes"] = 0
            return float(accuracy), len(self.X), metrics
        except Exception as e:
            print(f"Client {self.client_id} - Evaluation error: {str(e)}")
            return 0.0, len(self.X), {"accuracy": 0.0, "has_both_classes": 0}

def main():
    parser = argparse.ArgumentParser(description="Flower client with DP")
    parser.add_argument("--client-id", type=int, default=0)
    args = parser.parse_args()
    client = NetworkIntrusionClient(client_id=args.client_id)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)

if __name__ == "__main__":
    main()
