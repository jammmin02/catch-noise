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

# MLflow 설정 (공용 서버 주소)
mlflow.set_tracking_uri("http://210.101.236.174:5000")
mlflow.set_experiment("optuna_cnn_2class")

# 폴더 경로 설정 (robust_v7 전처리 기준)
base_dir = "/app/dev/jungmin/2class_noisy_vs_nonnoisy/cnn_only_v1/outputs"

# 최적 하이퍼파라미터 로드
with open(os.path.join(base_dir, "best_params.json"), "r") as f:
    best_params = json.load(f)

# 데이터 로드 (정규화 scaler 포함)
batch_size = best_params["batch_size"]
_, _, test_loader = load_data(batch_size=batch_size, base_dir=base_dir)

# 입력 shape 자동 감지
X_sample = np.load(os.path.join(base_dir, "X_cnn.npy"))
input_shape = X_sample.shape[1:]  # (time_steps, n_features)

# CNN 모델 정의
model = CNNOnly(
    input_shape=input_shape,
    conv1_filters=best_params["conv1_filters"],
    conv2_filters=best_params["conv2_filters"],
    dense_units=best_params["dense_units"],
    dropout=best_params["dropout"]
).to(device)

# 학습된 모델 파라미터 로드
model_path = os.path.join(base_dir, "best_model.pt")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 테스트셋 평가 수행
y_true, y_pred = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        yb = yb.view(-1, 1)

        logits = model(xb)
        probs = torch.sigmoid(logits).cpu().numpy().squeeze()

        targets_batch = yb.cpu().numpy().squeeze()
        preds_batch = (probs > 0.5).astype(int)

        y_pred.extend(preds_batch.tolist())
        y_true.extend(targets_batch.tolist())

# 최종 테스트 정확도
acc = accuracy_score(y_true, y_pred)
print(f"Test Accuracy: {acc:.4f}")

# MLflow 기록
with mlflow.start_run(run_name="evaluate_best_model"):
    mlflow.log_metric("test_accuracy", acc)
    mlflow.log_params(best_params)
    mlflow.log_artifact(model_path)
    print(f"모델 평가 및 MLflow 기록 완료 → {model_path}")
