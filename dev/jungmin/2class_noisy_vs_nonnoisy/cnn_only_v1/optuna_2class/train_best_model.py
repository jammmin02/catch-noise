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

# 디바이스 설정 (GPU 사용 가능 시 GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MLflow 설정 (서버 주소는 팀 공용으로 관리됨)
mlflow.set_tracking_uri("http://210.101.236.174:5000")
mlflow.set_experiment("optuna_cnn_2class")

# 최적 하이퍼파라미터 로드
with open("outputs/cnn_only/best_params.json", "r") as f:
    best_params = json.load(f)

# 데이터 로드 (이번엔 테스트셋 평가)
batch_size = best_params["batch_size"]
_, _, test_loader = load_data(batch_size)

# CNN 모델 정의 (input_shape 명확히 명시)
input_shape = (86, 14)
model = CNNOnly(
    input_shape=input_shape,
    conv1_filters=best_params["conv1_filters"],
    conv2_filters=best_params["conv2_filters"],
    dense_units=best_params["dense_units"],
    dropout=best_params["dropout"]
).to(device)

# 학습 완료된 모델 파라미터 로드
model_path = "outputs/cnn_only/best_model.pt"
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

# 최종 테스트 정확도 계산
acc = accuracy_score(y_true, y_pred)
print(f"Test Accuracy: {acc:.4f}")

# MLflow 기록 (최종 평가 결과)
with mlflow.start_run(run_name="evaluate_best_model"):
    mlflow.log_metric("test_accuracy", acc)
    mlflow.log_params(best_params)
    mlflow.log_artifact(model_path)
    print(f"모델 평가 및 로그 완료 → {model_path}")
