import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

# Define column names for the KDD Cup 99 dataset (10% subset)
COLUMN_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
    'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
    'num_outbound_cmds', 'is_host_login', 'is_guest_login',
    'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'class'
]


def load_and_preprocess_data(file_path):
    # Load the dataset from a local file (ensure the file is present)
    data = pd.read_csv(file_path, header=None, names=COLUMN_NAMES)

    # One-hot encode categorical features
    data = pd.get_dummies(data, columns=['protocol_type', 'service', 'flag'])

    # Normalize selected numerical features (you can extend this to more features as needed)
    numerical_cols = ['duration', 'src_bytes', 'dst_bytes', 'count', 'srv_count']
    scaler = MinMaxScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    # Map class labels to numeric codes
    data['class'] = data['class'].astype('category').cat.codes

    # Separate features and labels
    X = data.drop('class', axis=1).values.astype(np.float32)
    y = data['class'].values.astype(np.int64)
    return X, y


def create_dataloaders(X, y, batch_size=64, test_size=0.3):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# Define a simple neural network for intrusion detection
class IntrusionNet(nn.Module):
    def _init_(self, input_dim, num_classes):
        super(IntrusionNet, self)._init_()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Define the Flower client
class IntrusionClient(fl.client.NumPyClient):
    def _init_(self, model, train_loader, test_loader, device):
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
        # Train for one epoch (for demo purposes)
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
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
        return loss, len(self.test_loader.dataset), {"accuracy": accuracy}


if _name_ == "_main_":
    # Update the file path to your local copy of the dataset
    file_path = "kddcup.data_10_percent.gz"
    X, y = load_and_preprocess_data(file_path)

    train_loader, test_loader = create_dataloaders(X, y, batch_size=64)

    input_dim = X.shape[1]
    num_classes = len(np.unique(y))
    model = IntrusionNet(input_dim, num_classes)

    device = torch.device("cpu")  # Change to "cuda" if GPU is available
    model.to(device)

    # Start the Flower client; update server IP address accordingly
    fl.client.start_numpy_client(server_address="192.168.1.100:8080",
                                 client=IntrusionClient(model, train_loader, test_loader, device))