import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import accuracy_score
import mlflow
import time
from model import CNNLSTM

# ✅ Trial별 독립적인 MLflow Run 생성 + 최소 기록
def safe_log_mlflow(trial, acc, max_retries=3, wait_sec=1):
    acc = round(acc, 4)
    for attempt in range(max_retries):
        try:
            with mlflow.start_run(run_name=f"trial_{trial.number}"):  # ✅ 개별 run
                for k, v in trial.params.items():
                    mlflow.log_param(k, v)
                mlflow.log_metric("val_accuracy", acc)
            time.sleep(wait_sec)
            return True
        except Exception as e:
            print(f"[⚠️ MLflow 기록 실패 - Trial {trial.number}, 시도 {attempt+1}] {e}")
            time.sleep(2)
    print(f"❌ Trial {trial.number} val_accuracy 기록 실패")
    return False

# ✅ Optuna용 objective 함수 (바뀐 구조 반영)
def objective(trial, device, X_train, X_val):
    # 🔧 하이퍼파라미터 탐색
    conv1 = trial.suggest_categorical("conv1_filters", [16, 32, 64])
    conv2 = trial.suggest_categorical("conv2_filters", [32, 64, 128])
    lstm_units = trial.suggest_int("lstm_units", 32, 128, step=32)
    dense_units = trial.suggest_int("dense_units", 32, 128, step=32)
    dropout = trial.suggest_float("dropout", 0.2, 0.5, step=0.1)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    # ✅ 바뀐 구조에 맞게 CNNLSTM 생성 (timesteps, features 제거됨)
    model = CNNLSTM(
        conv1_filters=conv1,
        conv2_filters=conv2,
        lstm_units=lstm_units,
        dense_units=dense_units,
        dropout=dropout
    ).to(device)

    loss_fn = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=lr)

    # ✅ 학습
    for epoch in range(10):
        model.train()
        for xb, yb in X_train:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = loss_fn(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # ✅ 평가
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in X_val:
            xb = xb.to(device)
            output = model(xb)
            pred_np = (output.cpu().detach().numpy() > 0.5).astype(int).tolist()
            target_np = yb.numpy().astype(int).tolist()
            preds.extend(pred_np)
            targets.extend(target_np)

    acc = accuracy_score(targets, preds)
    safe_log_mlflow(trial, acc)
    return 1.0 - acc  # minimize
