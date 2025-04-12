# client.py
import os
import numpy as np
import pandas as pd
import flwr as fl
import joblib
from sklearn.preprocessing import RobustScaler
import argparse

class NetworkIntrusionClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        # Load client data
        self.client_id = client_id
        client_file = f'client_{client_id}.parquet'
        
        # Check if file exists
        if not os.path.exists(client_file):
            raise FileNotFoundError(f"Client data not found: {client_file}")
            
        # Load data
        self.df = pd.read_parquet(client_file)
        
        # Prepare features and labels
        self.X = self.df.drop(columns=['Label'])
        self.y = pd.get_dummies(self.df['Label'])  # One-hot encode labels
        
        # Load scaler if available
        if os.path.exists("robust_scaler.pkl"):
            self.scaler = joblib.load("robust_scaler.pkl")
        else:
            print("Warning: Scaler not found")
    
    def get_parameters(self, config):
        # Get current model parameters
        # In feature selection case, this would be your feature mask
        return self.get_feature_mask()
        
    def get_feature_mask(self):
        # If you don't have a stored feature mask, create a default one
        # This is a placeholder - your actual implementation might use chimp_optimization
        columns = self.X.columns
        feature_mask = np.ones(len(columns), dtype=np.int32)
        return [feature_mask]
    
    def fit(self, parameters, config):
        # Apply received parameters (feature mask)
        feature_mask = parameters[0]
        
        # Get selected features based on mask
        selected_features = np.where(feature_mask == 1)[0]
        X_selected = self.X.iloc[:, selected_features]
        
        # Your local training here
        print(f"Client {self.client_id}: Training with {len(selected_features)} features")
        
        # Return updated parameters, sample size, and metrics
        updated_mask = self.run_local_optimization(X_selected)
        return [updated_mask], len(self.X), {"client_id": self.client_id}
    
    
    def run_local_optimization(self, X):
    # Handle the case when only one feature is left
        if X.shape[1] <= 1:
            mask = np.zeros(len(self.X.columns), dtype=np.int32)
            # Keep the current feature selected
            selected_features = np.where(self.X.columns.isin(X.columns))[0]

            if len(selected_features) > 0:
                mask[selected_features[0]] = 1
            return mask
            
        # Original code for multiple features
        importance = np.abs(np.corrcoef(X.T))[0]
        mask = np.zeros(len(self.X.columns), dtype=np.int32)
        mask[np.argsort(importance)[-int(len(importance)*0.5):]] = 1
        return mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-id", type=int, default=0)
    parser.add_argument("--server", type=str, default="127.0.0.1:8080")
    args = parser.parse_args()
    
    # Start client
    fl.client.start_numpy_client(
        server_address=args.server,
        client=NetworkIntrusionClient(client_id=args.client_id)
    )

if __name__ == "__main__":
    main()