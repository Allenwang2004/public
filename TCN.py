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

# ====== 3. TCN 模型 ======
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        return out + self.downsample(x)  # 殘差連結

class TCNModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # ===== 可調參數 =====
        num_channels = [32, 64]     # 每層的輸出通道數（可以調整層數與寬度）
        kernel_size = 3             # 卷積核大小（可嘗試 2, 3, 5）
        dilations = [1, 2]          # 膨脹率，每層可設不同的 dilation

        layers = []
        in_channels = input_dim
        for out_channels, dilation in zip(num_channels, dilations):
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, dilation))
            in_channels = out_channels

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)

    def forward(self, x):  # x shape: (B, T, F)
        x = x.transpose(1, 2)  # -> (B, F, T)
        out = self.network(x)
        return self.fc(out[:, :, -1])

# ====== 4. 訓練流程 ======
def train_with_window_size(X_train_raw, y_train_raw, X_valid_raw, y_valid_raw, window_size, save_dir="models_tcn"):
    os.makedirs(save_dir, exist_ok=True)
    results = []

    X_train, y_train = create_sliding_window_data(X_train_raw, y_train_raw, window_size)
    X_valid, y_valid = create_sliding_window_data(X_valid_raw, y_valid_raw, window_size)

    train_ds = TimeSeriesDataset(X_train, y_train)
    valid_ds = TimeSeriesDataset(X_valid, y_valid)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=32)

    model = TCNModel(input_dim=X_train.shape[2])
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
            torch.save(model.state_dict(), f"{save_dir}/tcn_window{window_size}.pt")
        else:
            counter += 1
            if counter >= patience:
                tqdm.write("Early stopping triggered.")
                break

    model.load_state_dict(torch.load(f"{save_dir}/tcn_window{window_size}.pt"))
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

    pd.DataFrame(all_results).to_csv("tcn_window_comparison.csv", index=False)
    print("TCN results saved to tcn_window_comparison.csv")

