import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import os
import pandas as pd
import math

# ====== 1. Sliding window dataset ======
def create_sliding_window_data(X, y, window_size):
    X_seq, y_seq = [], []
    for i in range(len(X) - window_size):
        X_seq.append(X[i:i+window_size])
        y_seq.append(y[i+window_size])
    return np.array(X_seq), np.array(y_seq).reshape(-1, 1)

# ====== 2. Dataset Class ======
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ====== 3. Positional Encoding ======
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):  # x: (B, T, D)
        x = x + self.pe[:, :x.size(1)]
        return x

# ====== 4. Transformer 模型 ======
class TransformerModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # ===== 可調參數 =====
        d_model = 64          # 編碼器輸出維度（Transformer 隱藏維度）
        nhead = 4             # 注意力頭數量（多頭注意力）
        num_layers = 2        # 編碼器層數（深度）
        dim_feedforward = 128 # FeedForward 網路的中間層維度
        dropout = 0.1         # Dropout 比例，防止 overfitting

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):  # x: (B, T, D)
        x = self.input_proj(x)       # (B, T, d_model)
        x = self.pos_encoder(x)
        out = self.transformer_encoder(x)  # (B, T, d_model)
        return self.fc(out[:, -1, :])      # 取最後一個時間點做預測

# ====== 5. 訓練流程 ======
def train_with_window_size(X_train_raw, y_train_raw, X_valid_raw, y_valid_raw, window_size, save_dir="models_transformer"):
    os.makedirs(save_dir, exist_ok=True)
    results = []

    X_train, y_train = create_sliding_window_data(X_train_raw, y_train_raw, window_size)
    X_valid, y_valid = create_sliding_window_data(X_valid_raw, y_valid_raw, window_size)

    train_ds = TimeSeriesDataset(X_train, y_train)
    valid_ds = TimeSeriesDataset(X_valid, y_valid)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=32)

    model = TransformerModel(input_dim=X_train.shape[2])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_loss = float('inf')
    patience, counter = 5, 0

    for epoch in tqdm(range(50), desc=f"Training window_size={window_size}"):
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
            torch.save(model.state_dict(), f"{save_dir}/transformer_window{window_size}.pt")
        else:
            counter += 1
            if counter >= patience:
                tqdm.write("Early stopping triggered.")
                break

    model.load_state_dict(torch.load(f"{save_dir}/transformer_window{window_size}.pt"))
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in valid_dl:
            pred = model(xb)
            preds.append(pred.numpy())
            trues.append(yb.numpy())

    y_pred = np.vstack(preds)
    y_true = np.vstack(trues)

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    tqdm.write(f"Final MSE: {mse:.4f} | MAE: {mae:.4f}\n")

    results.append({"window_size": window_size, "mse": mse, "mae": mae})
    return results

# ====== Example usage ======
if __name__ == '__main__':
    np.random.seed(0)
    X_all = np.random.randn(1000, 10)
    y_all = np.random.randn(1000)

    X_train_raw = X_all[:800]
    y_train_raw = y_all[:800]
    X_valid_raw = X_all[800:]
    y_valid_raw = y_all[800:]

    all_results = []
    for win_size in [10, 20, 30]:
        result = train_with_window_size(X_train_raw, y_train_raw, X_valid_raw, y_valid_raw, window_size=win_size)
        all_results.extend(result)

    pd.DataFrame(all_results).to_csv("transformer_window_comparison.csv", index=False)
    print("Transformer results saved to transformer_window_comparison.csv")
