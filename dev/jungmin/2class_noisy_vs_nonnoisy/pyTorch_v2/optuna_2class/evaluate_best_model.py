import os
import json
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix
from data_loader import load_data
from model import CNNLSTM
torch.backends.cudnn.enabled = False # CUDA 비활성화

# ✅ 환경 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "outputs/cnn_lstm/best_model.pt"
param_path = "outputs/cnn_lstm/best_params.json"

# ✅ 최적 파라미터 로드
with open(param_path, "r") as f:
    best_params = json.load(f)

batch_size = best_params["batch_size"]

# ✅ 테스트 데이터만 로드
_, _, test_loader = load_data(batch_size)

# ✅ 모델 구성 및 가중치 로드
model = CNNLSTM(
    conv1_filters=best_params["conv1_filters"],
    conv2_filters=best_params["conv2_filters"],
    lstm_units=best_params["lstm_units"],
    dense_units=best_params["dense_units"],
    dropout=best_params["dropout"]
).to(device)

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ✅ 예측 수행
all_preds = []
all_targets = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        preds = model(xb).cpu().numpy()
        preds_bin = (preds > 0.5).astype(int).flatten().tolist()
        targets_bin = yb.numpy().astype(int).flatten().tolist()
        all_preds.extend(preds_bin)
        all_targets.extend(targets_bin)

# ✅ 평가 결과 출력
acc = accuracy_score(all_targets, all_preds)
cm = confusion_matrix(all_targets, all_preds)

print(f"\n✅ [Test Accuracy]: {acc:.4f}")
print("\n🧾 [Confusion Matrix]")
print(f"        Pred 0    Pred 1")
print(f"True 0    {cm[0][0]:>6}     {cm[0][1]:>6}")
print(f"True 1    {cm[1][0]:>6}     {cm[1][1]:>6}")
