import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import pandas as pd
import numpy as np
from opacus import PrivacyEngine
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
import base64

class SimpleHomomorphicEncryptor:
    def __init__(self):
        self.key = RSA.generate(2048)
        self.pub_key = self.key.publickey()
        self.encryptor = PKCS1_OAEP.new(self.pub_key)
        self.decryptor = PKCS1_OAEP.new(self.key)

    def encrypt_batch(self, batch):
        # Simulate encryption (not true homomorphic but demonstrative)
        encrypted = []
        for row in batch:
            enc_row = []
            for val in row:
                val_str = f"{val:.4f}"
                enc = self.encryptor.encrypt(val_str.encode())
                enc_row.append(enc)
            encrypted.append(enc_row)
        return encrypted

    def decrypt_batch(self, encrypted_batch):
        decrypted = []
        for row in encrypted_batch:
            dec_row = []
            for enc in row:
                try:
                    dec = self.decryptor.decrypt(enc)
                    dec_val = float(dec.decode())
                    dec_row.append(dec_val)
                except:
                    dec_row.append(0.0)
            decrypted.append(dec_row)
        return torch.tensor(decrypted, dtype=torch.float32)

# ========== CONFIG ========== #
EPOCHS = 5
BATCH_SIZE = 64
USE_DIFFERENTIAL_PRIVACY = True
DP_NOISE_MULTIPLIER = 1.0
MAX_GRAD_NORM = 1.0
DATA_PATH = "02-14-2018.txt"

# ========== LOAD & PREPROCESS ========== #
def load_data(file_path):
    data = pd.read_csv(file_path)
    data = data.drop(columns=["Timestamp"])
    selected_features = [
        'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
        'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Flow Byts/s',
        'Flow Pkts/s', 'Fwd IAT Mean', 'Bwd IAT Mean'
    ]
    data = data[selected_features + ['Label']]
    data[data.select_dtypes(include=[np.number]).columns] = data.select_dtypes(include=[np.number]).apply(
        lambda x: np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
    )
    data["Label"] = LabelEncoder().fit_transform(data["Label"])
    X = MinMaxScaler().fit_transform(data.iloc[:, :-1])
    y = data.iloc[:, -1].values
    return X.astype(np.float32), y.astype(np.int64)

# ========== MODEL ========== #
class IntrusionFNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(IntrusionFNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

# ========== TRAIN ========== #
def train_model(model, train_loader, test_loader, use_dp=False, use_he=False):
    device = torch.device("cpu")
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    encryptor = SimpleHomomorphicEncryptor() if use_he else None

    # Apply DP if requested
    if use_dp:
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=DP_NOISE_MULTIPLIER,
            max_grad_norm=MAX_GRAD_NORM,
        )
        print(f"[+] Differential Privacy ENABLED (ε will be calculated after training).")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            if use_he:
                enc_data = encryptor.encrypt_batch(data.tolist())
                data = encryptor.decrypt_batch(enc_data).to(device)  # simulate computation on encrypted data

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.size(0)
        avg_loss = total_loss / len(train_loader.dataset)

        # Evaluate
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
        accuracy = correct / len(test_loader.dataset)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")

    if use_dp:
        epsilon = privacy_engine.get_epsilon(delta=1e-5)
        print(f"[DP STATS] ε (epsilon): {epsilon:.2f}, δ (delta): 1e-5, Noise Multiplier: {DP_NOISE_MULTIPLIER}")

    return accuracy

# ========== MAIN ========== #
if __name__ == "__main__":
    X, y = load_data(DATA_PATH)
    X1, X2, y1, y2 = train_test_split(X, y, test_size=0.5, random_state=42)

    print("\n[Client 1] Training WITHOUT Differential Privacy...")
    loader1 = DataLoader(TensorDataset(torch.tensor(X1), torch.tensor(y1)), batch_size=BATCH_SIZE, shuffle=True)
    test_loader1 = DataLoader(TensorDataset(torch.tensor(X2), torch.tensor(y2)), batch_size=BATCH_SIZE)
    model1 = IntrusionFNN(X.shape[1], len(np.unique(y)))
    acc1 = train_model(model1, loader1, test_loader1, use_dp=False)

    print("\n[Client 2] Training WITH Differential Privacy...")
    loader2 = DataLoader(TensorDataset(torch.tensor(X1), torch.tensor(y1)), batch_size=BATCH_SIZE, shuffle=True)
    test_loader2 = DataLoader(TensorDataset(torch.tensor(X2), torch.tensor(y2)), batch_size=BATCH_SIZE)
    model2 = IntrusionFNN(X.shape[1], len(np.unique(y)))
    acc2 = train_model(model2, loader2, test_loader2, use_dp=True)

    print("\n[Client 3] Training WITH Homomorphic Encryption...")
    model3 = IntrusionFNN(X.shape[1], len(np.unique(y)))
    acc3 = train_model(model3, loader1, test_loader1, use_dp=False, use_he=True)

    print("\n[Client 4] Training WITH Differential Privacy + Homomorphic Encryption...")
    model4 = IntrusionFNN(X.shape[1], len(np.unique(y)))
    acc4 = train_model(model4, loader1, test_loader1, use_dp=True, use_he=True)

    print("\n======= FINAL COMPARISON =======")
    print(f"1. No Privacy             - Accuracy: {acc1:.4f}")
    print(f"2. Differential Privacy   - Accuracy: {acc2:.4f}")
    print(f"3. Homomorphic Encryption - Accuracy: {acc3:.4f}")
    print(f"4. DP + HE                - Accuracy: {acc4:.4f}")
