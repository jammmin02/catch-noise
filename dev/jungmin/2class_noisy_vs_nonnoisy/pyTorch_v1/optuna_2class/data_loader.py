import os
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch

def load_data(batch_size):
    base_dir = "/app/outputs/cnn_lstm"  # ✅ Docker 기준 절대경로

    X_path = os.path.join(base_dir, "X_lstm.npy")
    y_path = os.path.join(base_dir, "y_lstm.npy")

    # ✅ Numpy 파일 로드
    X = np.load(X_path)  # [B, 86, 14]
    y = np.load(y_path)

    # ✅ 채널 차원 추가 (→ [B, 1, 86, 14])
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    # ✅ 데이터 분할 (7:2:1)
    total = len(X_tensor)
    train_end = int(total * 0.7)
    val_end   = int(total * 0.9)

    train_dataset = TensorDataset(X_tensor[:train_end], y_tensor[:train_end])
    val_dataset   = TensorDataset(X_tensor[train_end:val_end], y_tensor[train_end:val_end])
    test_dataset  = TensorDataset(X_tensor[val_end:], y_tensor[val_end:])

    # ✅ DataLoader 구성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size)

    # ✅ timesteps, features는 현재 구조에서 사용 안 함 (CNN+LSTM이 자동 처리)
    return train_loader, val_loader, test_loader, None, None