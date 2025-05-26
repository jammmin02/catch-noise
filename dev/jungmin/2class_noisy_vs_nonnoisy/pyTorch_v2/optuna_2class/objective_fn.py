import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import accuracy_score
import mlflow
import time
from model import CNNLSTM
from data_loader import load_data

# ✅ MLflow 안전 로그 함수
def safe_log_mlflow(trial, acc, max_retries=3, wait_sec=1):
    acc = round(acc, 4)
    for attempt in range(max_retries):
        try:
            with mlflow.start_run(run_name=f"trial_{trial.number}"):
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

# ✅ Optuna objective 함수
def objective(trial, device):
    try:
        # 🔧 하이퍼파라미터 탐색
        conv1 = trial.suggest_categorical("conv1_filters", [16, 32, 64])
        conv2 = trial.suggest_categorical("conv2_filters", [32, 64, 128])
        lstm_units = trial.suggest_int("lstm_units", 32, 128, step=32)
        dense_units = trial.suggest_int("dense_units", 32, 128, step=32)
        dropout = trial.suggest_float("dropout", 0.2, 0.5, step=0.1)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        print(f"\n🚀 [Trial {trial.number}] 시작합니다.")
        print(f"🔧 하이퍼파라미터: conv1={conv1}, conv2={conv2}, lstm={lstm_units}, dense={dense_units}, dropout={dropout}, lr={lr:.5f}, batch_size={batch_size}")

        # ✅ 데이터 로딩
        train_loader, val_loader, _ = load_data(batch_size)

        # ✅ 모델 정의
        model = CNNLSTM(conv1, conv2, lstm_units, dense_units, dropout).to(device)
        loss_fn = nn.BCELoss()
        optimizer = Adam(model.parameters(), lr=lr)

        # ✅ 학습
        for epoch in range(1, 11):
            model.train()
            running_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                if xb.dim() == 3:  # 채널 차원 없으면 추가
                    xb = xb.unsqueeze(1)
                yb = yb.view(-1, 1)

                preds = model(xb)
                loss = loss_fn(preds, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            avg_loss = running_loss / len(train_loader)
            print(f"📘 Epoch {epoch:2d}/10 - 평균 Loss: {avg_loss:.4f}")

        # ✅ 검증
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                if xb.dim() == 3:
                    xb = xb.unsqueeze(1)
                yb = yb.view(-1, 1)

                output = model(xb).cpu().squeeze().numpy()
                target_np = yb.cpu().squeeze().numpy()
                pred_np = (output > 0.5).astype(int).tolist()
                target_np = target_np.astype(int).tolist()

                preds.extend(pred_np)
                targets.extend(target_np)

        acc = accuracy_score(targets, preds)
        print(f"✅ [Trial {trial.number}] 완료 - val_accuracy: {acc:.4f}")

        # ✅ MLflow 로그
        safe_log_mlflow(trial, acc)
        return 1.0 - acc

    except Exception as e:
        print(f"❌ Trial {trial.number} 실패: {str(e)}")
        return float("inf")
