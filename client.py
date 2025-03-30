import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from torch.utils.data import TensorDataset, DataLoader

# -----------------------------
# Data Loading and Preprocessing
# -----------------------------
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    
    # Drop the non-numeric column 'Timestamp'
    if 'Timestamp' in data.columns:
        data = data.drop(columns=["Timestamp"])
    
    # Select a subset of useful features for intrusion detection
    selected_features = [
        'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 
        'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Flow Byts/s', 
        'Flow Pkts/s', 'Fwd IAT Mean', 'Bwd IAT Mean'
    ]
    data = data[selected_features + ['Label']]
    
    # Replace non-finite values in numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].apply(
        lambda x: np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
    )
    
    # Encode categorical labels (assuming 'Label' is the target)
    label_encoder = LabelEncoder()
    data["Label"] = label_encoder.fit_transform(data["Label"])
    
    # Separate features and target
    X = data.iloc[:, :-1].values.astype(np.float32)
    y = data.iloc[:, -1].values.astype(np.int64)
    
    # Normalize features
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    return X, y

# -----------------------------
# Create DataLoaders
# -----------------------------
def create_dataloaders(X, y, batch_size=64, test_size=0.3):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# -----------------------------
# Define a Feedforward Neural Network (FNN)
# -----------------------------
class IntrusionFNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(IntrusionFNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(p=0.3)  # Regularization
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# -----------------------------
# Define the Flower Client
# -----------------------------
class IntrusionClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = self.model.state_dict()
        for key, val in zip(params_dict.keys(), parameters):
            params_dict[key] = torch.tensor(val)
        self.model.load_state_dict(params_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for data, target in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
        print("Local training complete.")
        return self.get_parameters({}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss, correct = 0, 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss += self.criterion(output, target).item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        loss /= len(self.test_loader.dataset)
        accuracy = correct / len(self.test_loader.dataset)
        print(f"Evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        return loss, len(self.test_loader.dataset), {"accuracy": accuracy}

# -----------------------------
# Main Execution Block
# -----------------------------
if __name__ == "__main__":
    file_path = "Fedrated_Privacy_Proj/02-14-2018.csv"  # Update with your dataset path
    X, y = load_and_preprocess_data(file_path)
    train_loader, test_loader = create_dataloaders(X, y, batch_size=64)
    
    input_dim = X.shape[1]
    num_classes = len(np.unique(y))
    model = IntrusionFNN(input_dim, num_classes)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Local Training and Evaluation Block
    num_epochs = 30
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    print("Starting local training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * data.size(0)
        running_loss /= len(train_loader.dataset)
        
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += data.size(0)
        accuracy = correct / total
        print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {running_loss:.4f}, Test Accuracy: {accuracy:.4f}")
    
    # Uncomment for federated learning
    # fl.client.start_numpy_client(server_address="192.168.1.100:8080", client=IntrusionClient(model, train_loader, test_loader, device))
