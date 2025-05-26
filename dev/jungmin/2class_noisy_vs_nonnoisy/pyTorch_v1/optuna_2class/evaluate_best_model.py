import os
import json
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix
from data_loader import load_data
from model import CNNLSTM

# ✅ 환경 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "best_model.pt"

# ✅ 저장된 파라미터 로드
with open("best_params.json", "r") as f:
    best_params = json.load(f)

batch_size = best_params["batch_size"]
_, _, test_loader, _, _ = load_data(batch_size)  # timesteps, features 제거됨

# ✅ 모델 초기화 및 가중치 불러오기 (새 구조 기준)
model = CNNLSTM(
    conv1_filters=best_params["conv1_filters"],
    conv2_filters=best_params["conv2_filters"],
    lstm_units=best_params["lstm_units"],
    dense_units=best_params["dense_units"],
    dropout=best_params["dropout"]
).to(device)

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ✅ 테스트 평가
all_preds = []
all_targets = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        preds = model(xb).cpu().numpy()
        all_preds.extend((preds > 0.5).astype(int).flatten().tolist())
        all_targets.extend(yb.numpy().flatten().astype(int).tolist())

# ✅ 출력
acc = accuracy_score(all_targets, all_preds)
cm = confusion_matrix(all_targets, all_preds)

print(f"\n✅ Test Accuracy: {acc:.4f}")
print("🧾 Confusion Matrix:")
print(cm)
