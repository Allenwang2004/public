import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import os
import pandas as pd

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

# ====== 3. LSTM + CNN 模型 ======
class LSTMCNNModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # ===== 可調參數 =====
        hidden_dim = 64          # LSTM 隱藏層維度
        num_layers = 2           # LSTM 疊加層數
        dropout = 0.3            # LSTM dropout 比例
        cnn_out_channels = 32    # CNN 輸出通道數（feature map 數量）
        cnn_kernel_size = 3      # CNN 卷積核大小（控制局部特徵範圍）

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        self.cnn = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=cnn_out_channels,
            kernel_size=cnn_kernel_size
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(cnn_out_channels, 1)

    def forward(self, x):  # x: (B, T, F)
        lstm_out, _ = self.lstm(x)          # (B, T, H)
        cnn_in = lstm_out.transpose(1, 2)   # (B, H, T)
        cnn_out = self.cnn(cnn_in)          # (B, C, T-k+1)
        pooled = self.pool(cnn_out).squeeze(-1)  # (B, C)
        return self.fc(pooled)              # (B, 1)

# ====== 4. 訓練流程 ======
def train_with_window_size(X_train_raw, y_train_raw, X_valid_raw, y_valid_raw, window_size, save_dir="models_lstmc"):
    os.makedirs(save_dir, exist_ok=True)
    results = []

    X_train, y_train = create_sliding_window_data(X_train_raw, y_train_raw, window_size)
    X_valid, y_valid = create_sliding_window_data(X_valid_raw, y_valid_raw, window_size)

    train_ds = TimeSeriesDataset(X_train, y_train)
    valid_ds = TimeSeriesDataset(X_valid, y_valid)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=32)

    model = LSTMCNNModel(input_dim=X_train.shape[2])
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
            torch.save(model.state_dict(), f"{save_dir}/lstmc_window{window_size}.pt")
        else:
            counter += 1
            if counter >= patience:
                tqdm.write("Early stopping triggered.")
                break

    model.load_state_dict(torch.load(f"{save_dir}/lstmc_window{window_size}.pt"))
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

    pd.DataFrame(all_results).to_csv("lstmc_window_comparison.csv", index=False)
    print("LSTM+CNN results saved to lstmc_window_comparison.csv")
