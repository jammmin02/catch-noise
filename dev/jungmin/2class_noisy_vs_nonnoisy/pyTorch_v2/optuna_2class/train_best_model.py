import os
import json
import torch
import numpy as np
import mlflow
from sklearn.metrics import accuracy_score
from torch.nn import BCELoss
from torch.optim import Adam

from model import CNNLSTM
from data_loader import load_data  # stratified 기반 분할

# ✅ 환경 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ MLflow 설정
mlflow.set_tracking_uri("http://210.101.236.174:5000")
mlflow.set_experiment("optuna_cnn_lstm_2class")

# ✅ 최적 파라미터 로드
with open("best_params.json", "r") as f:
    best_params = json.load(f)

# ✅ 데이터 로드 (test set 평가 목적)
batch_size = best_params["batch_size"]
_, _, test_loader = load_data(batch_size)

# ✅ 모델 초기화 및 파라미터 적용
model = CNNLSTM(
    conv1_filters=best_params["conv1_filters"],
    conv2_filters=best_params["conv2_filters"],
    lstm_units=best_params["lstm_units"],
    dense_units=best_params["dense_units"],
    dropout=best_params["dropout"]
).to(device)

model_path = "outputs/cnn_lstm/best_model.pt"
model.load_state_dict(torch.load(model_path))
model.eval()

# ✅ 평가 실행
y_true, y_pred = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        output = model(xb).cpu().squeeze().numpy()
        preds = (output > 0.5).astype(int).tolist()
        labels = yb.squeeze().numpy().astype(int).tolist()

        y_pred += preds
        y_true += labels

acc = accuracy_score(y_true, y_pred)
print(f"✅ Test Accuracy: {acc:.4f}")

# ✅ MLflow 기록
with mlflow.start_run(run_name="evaluate_best_model"):
    mlflow.log_metric("test_accuracy", acc)
    mlflow.log_params(best_params)
    mlflow.log_artifact(model_path)
    print(f"📦 평가된 모델 기록 완료: {model_path}")
