import os
import json
import torch
import numpy as np
import mlflow
from sklearn.metrics import accuracy_score
from model import CNNOnly
from data_loader import load_data

# ✅ 환경 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ MLflow 설정
mlflow.set_tracking_uri("http://210.101.236.174:5000")
mlflow.set_experiment("optuna_cnn_only_v2")  # v2 실험 이름으로 변경

# ✅ 경로 설정
BASE_DIR = "dev/jungmin/2class_noisy_vs_nonnoisy/cnn_only_v2"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# ✅ 최적 파라미터 로드
param_path = os.path.join(OUTPUT_DIR, "best_params.json")
with open(param_path, "r") as f:
    best_params = json.load(f)

# ✅ 데이터 로드 (test set 평가 목적)
batch_size = best_params["batch_size"]
_, _, test_loader = load_data(batch_size)

# ✅ 모델 초기화 및 파라미터 적용
model = CNNOnly(
    conv1_filters=best_params["conv1_filters"],
    conv2_filters=best_params["conv2_filters"],
    dense_units=best_params["dense_units"],
    dropout=best_params["dropout"]
).to(device)

model_path = os.path.join(OUTPUT_DIR, "best_model.pt")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ✅ 평가 실행
y_true, y_pred = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        output = model(xb).cpu()
        preds = torch.argmax(output, dim=1).numpy().tolist()
        labels = yb.cpu().numpy().tolist()

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
