import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import pandas as pd

# ====== 1. Dataset 不做 sliding window，直接用 tabular ======
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ====== 2. FCNN 模型（適用 tabular） ======
class FCNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

# ====== 3. FT-Transformer（簡化版） ======
class FTTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # (B, 1, d_model)
        out = self.transformer(x)           # (B, 1, d_model)
        return self.fc(out[:, 0, :])        # 取第一個 token 輸出

# ====== 4. 訓練流程 ======
def train_tabular_model(X_train, y_train, X_valid, y_valid, model_name="fcnn"):
    train_ds = TabularDataset(X_train, y_train)
    valid_ds = TabularDataset(X_valid, y_valid)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=32)

    input_dim = X_train.shape[1]
    if model_name == "fcnn":
        model = FCNN(input_dim)
        save_path = "fcnn_tabular.pt"
    elif model_name == "fttransformer":
        model = FTTransformer(input_dim)
        save_path = "fttransformer_tabular.pt"
    else:
        raise ValueError("Unsupported model_name")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_loss = float('inf')
    patience, counter = 5, 0

    for epoch in tqdm(range(50), desc=f"Training {model_name}"):
        model.train()
        train_loss = 0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)

        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for xb, yb in valid_dl:
                pred = model(xb)
                loss = criterion(pred, yb)
                valid_loss += loss.item() * len(xb)

        train_loss /= len(train_dl.dataset)
        valid_loss /= len(valid_dl.dataset)
        tqdm.write(f"Epoch {epoch+1:2d} | Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}")

        if valid_loss < best_loss:
            best_loss = valid_loss
            counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            counter += 1
            if counter >= patience:
                tqdm.write("Early stopping triggered.")
                break

    # 評估最佳模型
    model.load_state_dict(torch.load(save_path))
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in valid_dl:
            pred = model(xb)
            preds.append(pred.numpy())
            trues.append(yb.numpy())

    y_pred = np.vstack(preds)
    y_true = np.vstack(trues)

    print("Final MSE:", mean_squared_error(y_true, y_pred))
    print("Final MAE:", mean_absolute_error(y_true, y_pred))

# ====== 5. Example usage ======
if __name__ == '__main__':
    # 模擬 tabular data：1000 筆、10 特徵
    np.random.seed(0)
    X_all = np.random.randn(1000, 10)
    y_all = np.random.randn(1000)

    X_train = X_all[:800]
    y_train = y_all[:800]
    X_valid = X_all[800:]
    y_valid = y_all[800:]

    #train_tabular_model(X_train, y_train, X_valid, y_valid, model_name="fcnn")
    train_tabular_model(X_train, y_train, X_valid, y_valid, model_name="fttransformer")
