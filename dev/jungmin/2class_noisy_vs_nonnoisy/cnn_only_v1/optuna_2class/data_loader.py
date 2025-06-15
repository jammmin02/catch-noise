import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def stratified_split(X, y, seed=42):
    X_train_all, X_val_all, X_test_all = [], [], []
    y_train_all, y_val_all, y_test_all = [], [], []

    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        X_cls, y_cls = X[idx], y[idx]

        X_temp, X_test, y_temp, y_test = train_test_split(
            X_cls, y_cls, test_size=0.1, random_state=seed, shuffle=True
        )

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
    X_val, y_val     = concat_and_shuffle(X_val_all, y_val_all, seed=seed)
    X_test, y_test   = concat_and_shuffle(X_test_all, y_test_all, seed=seed)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def load_data(batch_size, base_dir=None, scaler_path=None, save_scaler=True):
    if base_dir is None:
        base_dir = "/app/dev/jungmin/2class_noisy_vs_nonnoisy/cnn_only_v1/outputs"

    X = np.load(os.path.join(base_dir, "X_cnn.npy"))
    y = np.load(os.path.join(base_dir, "y_cnn.npy"))

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = stratified_split(X, y)

    # 스케일링
    n_samples, time_steps, n_features = X_train.shape

    if scaler_path and os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        scaler = StandardScaler()
        scaler.fit(X_train.reshape(-1, n_features))
        if save_scaler:
            joblib.dump(scaler, os.path.join(base_dir, "scaler_cnn.pkl"))

    X_train_scaled = scaler.transform(X_train.reshape(-1, n_features)).reshape(n_samples, time_steps, n_features)
    X_val_scaled   = scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape[0], time_steps, n_features)
    X_test_scaled  = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape[0], time_steps, n_features)

    def to_tensor(X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # [B, 1, 시간, 특징]
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # [B, 1]
        return TensorDataset(X_tensor, y_tensor)

    train_dataset = to_tensor(X_train_scaled, y_train)
    val_dataset   = to_tensor(X_val_scaled, y_val)
    test_dataset  = to_tensor(X_test_scaled, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader
