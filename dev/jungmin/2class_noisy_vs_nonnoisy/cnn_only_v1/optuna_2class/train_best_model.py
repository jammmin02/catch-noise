import os
import json
import torch
import numpy as np
import mlflow
from sklearn.metrics import accuracy_score
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam

from model import CNNOnly
from data_loader import load_data

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MLflow 설정
mlflow.set_tracking_uri("http://210.101.236.174:5000")
mlflow.set_experiment("optuna_cnn_2class")

# 경로 설정
base_dir = "/app/dev/jungmin/2class_noisy_vs_nonnoisy/cnn_only_v1/outputs"

# Optuna 결과에서 best 하이퍼파라미터 로드
with open(os.path.join(base_dir, "best_params.json"), "r") as f:
    best_params = json.load(f)

# 전체 데이터를 로드하여 최종 재학습 진행 (train+val+test 전체)
X = np.load(os.path.join(base_dir, "X_cnn.npy"))
y = np.load(os.path.join(base_dir, "y_cnn.npy"))

# 스케일러 로드 및 전체 데이터 정규화
from sklearn.preprocessing import StandardScaler
import joblib

scaler = joblib.load(os.path.join(base_dir, "scaler_cnn.pkl"))
n_samples, time_steps, n_features = X.shape
X_scaled = scaler.transform(X.reshape(-1, n_features)).reshape(n_samples, time_steps, n_features)

# TensorDataset 준비
X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)

# DataLoader 준비
batch_size = best_params["batch_size"]
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 입력 shape 자동 감지
input_shape = X.shape[1:]  # (time_steps, n_features)

# CNN 모델 정의
model = CNNOnly(
    input_shape=input_shape,
    conv1_filters=best_params["conv1_filters"],
    conv2_filters=best_params["conv2_filters"],
    dense_units=best_params["dense_units"],
    dropout=best_params["dropout"]
).to(device)

# 손실함수 및 옵티마이저
loss_fn = BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=best_params["lr"])

# 최종 전체 재학습 진행
epochs = 20
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0.0
    for xb, yb in data_loader:
        xb, yb = xb.to(device), yb.to(device)
        yb = yb.view(-1, 1)

        logits = model(xb)
        loss = loss_fn(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    print(f"Epoch {epoch:2d} - Train Loss: {avg_loss:.4f}")

# 재학습 완료 → 전체 train accuracy 확인
model.eval()
all_preds, all_targets = [], []
with torch.no_grad():
    for xb, yb in data_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        probs = torch.sigmoid(logits).cpu().numpy().squeeze()
        targets = yb.cpu().numpy().squeeze()
        preds = (probs > 0.5).astype(int)
        all_preds.extend(preds.tolist())
        all_targets.extend(targets.tolist())

final_acc = accuracy_score(all_targets, all_preds)
print(f"Final Train Accuracy (on full data): {final_acc:.4f}")

# 모델 저장
model_path = os.path.join(base_dir, "best_model.pt")
torch.save(model.state_dict(), model_path)

# MLflow 기록 (최종 전체 재학습 기록)
with mlflow.start_run(run_name="final_retrain_full_data"):
    mlflow.log_metric("final_train_accuracy", final_acc)
    mlflow.log_params(best_params)
    mlflow.log_artifact(model_path)
    print(f"최종 모델 저장 및 MLflow 기록 완료 → {model_path}")
