import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

def stratified_split(X, y, seed=42):
    """
    클래스 비율을 유지한 채로 7:2:1로 데이터셋을 분할
    """
    X_train_all, X_val_all, X_test_all = [], [], []
    y_train_all, y_val_all, y_test_all = [], [], []

    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        X_cls, y_cls = X[idx], y[idx]

        # 1단계: temp 90%, test 10%
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_cls, y_cls, test_size=0.1, random_state=seed, shuffle=True
        )

        # 2단계: train 70%, val 20% (즉, temp 기준 7:2 비율)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=2/9, random_state=seed, shuffle=True
        )

        X_train_all.append(X_train)
        y_train_all.append(y_train)
        X_val_all.append(X_val)
        y_val_all.append(y_val)
        X_test_all.append(X_test)
        y_test_all.append(y_test)

    def concat_and_shuffle(X_parts, y_parts, seed=42):
        X_all = np.concatenate(X_parts)
        y_all = np.concatenate(y_parts)
        np.random.seed(seed)
        idx = np.random.permutation(len(X_all))
        return X_all[idx], y_all[idx]

    X_train, y_train = concat_and_shuffle(X_train_all, y_train_all, seed=seed)
    X_val, y_val = concat_and_shuffle(X_val_all, y_val_all, seed=seed)
    X_test, y_test = concat_and_shuffle(X_test_all, y_test_all, seed=seed)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def load_data(batch_size):
    """
    .npy 데이터를 로드하고 stratified 7:2:1로 나누어 DataLoader 반환
    """
    base_dir = "/app/dev/jungmin/2class_noisy_vs_nonnoisy/cnn_only_v1/outputs"
    X = np.load(os.path.join(base_dir, "X_cnn.npy"))  # [B, 86, 14]
    y = np.load(os.path.join(base_dir, "y_cnn.npy"))

    # stratified 분할
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = stratified_split(X, y)

    # 텐서 변환 (CNN을 위한 채널 차원 추가, 라벨은 BCE에 맞춰 float32 + unsqueeze)
    def to_tensor(X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # [B, 1, 86, 14]
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # [B, 1]
        return TensorDataset(X_tensor, y_tensor)

    train_dataset = to_tensor(X_train, y_train)
    val_dataset   = to_tensor(X_val, y_val)
    test_dataset  = to_tensor(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader
