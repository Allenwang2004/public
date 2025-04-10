import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import os
import pandas as pd

# ====== 1. 資料處理：滑動視窗轉換 ======
def create_sliding_window_data(X, y, window_size):
    X_seq, y_seq = [], []
    for i in range(len(X) - window_size):
        X_seq.append(X[i:i+window_size])
        y_seq.append(y[i+window_size])
    return np.array(X_seq), np.array(y_seq).reshape(-1, 1)

# ====== 2. 自訂 Dataset 類別 ======
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ====== 3. FCNN 模型定義 ======
class FCNN(nn.Module):
    def __init__(self, input_dim, seq_len):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.model(x)

# ====== 4. 主流程 ======
def train_with_window_size(X_train_raw, y_train_raw, X_valid_raw, y_valid_raw, window_size, save_dir="models"):
    os.makedirs(save_dir, exist_ok=True)
    results = []

    # Step 1: 製作時間序列資料
    X_train, y_train = create_sliding_window_data(X_train_raw, y_train_raw, window_size)
    X_valid, y_valid = create_sliding_window_data(X_valid_raw, y_valid_raw, window_size)

    # Step 2: DataLoader
    train_ds = TimeSeriesDataset(X_train, y_train)
    valid_ds = TimeSeriesDataset(X_valid, y_valid)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=32)

    # Step 3: 建立模型
    model = FCNN(input_dim=X_train.shape[2], seq_len=window_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # EarlyStopping 參數
    best_loss = float('inf')
    patience, counter = 5, 0

    # 記錄每 epoch loss
    train_losses, valid_losses = [], []

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
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        tqdm.write(f"Epoch {epoch+1:2d} | Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}")

        # Early stopping
        if valid_loss < best_loss:
            best_loss = valid_loss
            counter = 0
            torch.save(model.state_dict(), f"{save_dir}/fcnn_window{window_size}.pt")
        else:
            counter += 1
            if counter >= patience:
                tqdm.write("Early stopping triggered.")
                break

    # 最佳模型評估
    model.load_state_dict(torch.load(f"{save_dir}/fcnn_window{window_size}.pt"))
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
    return results, train_losses, valid_losses

# ====== Example Usage (需自行提供資料) ======
if __name__ == '__main__':
    # 模擬假資料：1000 時點、10 特徵
    np.random.seed(0)
    X_all = np.random.randn(1000, 10)
    y_all = np.random.randn(1000)

    # 切成訓練 / 驗證（前 800 為訓練）
    X_train_raw = X_all[:800]
    y_train_raw = y_all[:800]
    X_valid_raw = X_all[800:]
    y_valid_raw = y_all[800:]

    all_results = []
    for win_size in [10, 20, 30]:
        results, train_losses, valid_losses = train_with_window_size(
            X_train_raw, y_train_raw, X_valid_raw, y_valid_raw, window_size=win_size)
        all_results.extend(results)

    # 儲存結果
    df_result = pd.DataFrame(all_results)
    df_result.to_csv("fcnn_window_comparison.csv", index=False)
    print("所有結果已儲存到 fcnn_window_comparison.csv")
