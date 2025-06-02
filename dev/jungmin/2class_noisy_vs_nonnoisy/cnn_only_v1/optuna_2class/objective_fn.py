import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import accuracy_score
import mlflow
import time
from model import CNNOnly
from data_loader import load_data

def safe_log_mlflow(trial, acc, max_retries=3, wait_sec=1):
    """
    MLflow 기록 오류 발생 시 재시도하는 안전 로그 함수
    """
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
            print(f"[MLflow 기록 실패 - Trial {trial.number}, 시도 {attempt+1}] {e}")
            time.sleep(2)
    print(f"[Trial {trial.number}] val_accuracy 기록 실패")
    return False

def objective(trial, device):
    """
    Optuna 하이퍼파라미터 최적화를 위한 objective 함수
    """
    try:
        # 하이퍼파라미터 탐색 범위 설정
        conv1 = trial.suggest_categorical("conv1_filters", [16, 32, 64])
        conv2 = trial.suggest_categorical("conv2_filters", [32, 64, 128])
        dense_units = trial.suggest_int("dense_units", 32, 128, step=32)
        dropout = trial.suggest_float("dropout", 0.2, 0.5, step=0.1)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        print(f"\n[Trial {trial.number}] 시작")
        print(f"Params: conv1={conv1}, conv2={conv2}, dense={dense_units}, dropout={dropout}, lr={lr:.5f}, batch_size={batch_size}")

        # 데이터 로딩 (경로는 data_loader 내부에서 관리)
        train_loader, val_loader, _ = load_data(batch_size)

        # 입력 shape 정의 (전처리에서 고정)
        input_shape = (86, 14)
        model = CNNOnly(input_shape, conv1, conv2, dense_units, dropout).to(device)

        # 손실함수 및 옵티마이저 정의
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = Adam(model.parameters(), lr=lr)

        # 학습 루프 (고정 에폭 수: 10)
        for epoch in range(1, 11):
            model.train()
            total_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                yb = yb.view(-1, 1)

                logits = model(xb)
                loss = loss_fn(logits, yb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch:2d} - Train Loss: {avg_loss:.4f}")

        # 검증 (validation accuracy 계산)
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                yb = yb.view(-1, 1)

                logits = model(xb)
                probs = torch.sigmoid(logits).cpu().numpy().squeeze()

                targets_batch = yb.cpu().numpy().squeeze()
                preds_batch = (probs > 0.5).astype(int)

                preds.extend(preds_batch.tolist())
                targets.extend(targets_batch.tolist())

        acc = accuracy_score(targets, preds)
        print(f"[Trial {trial.number}] Validation Accuracy: {acc:.4f}")

        # MLflow 기록 (오류 방지 안전 기록)
        safe_log_mlflow(trial, acc)

        # Optuna 최소화 목표 → 정확도 반대로 반환
        return 1.0 - acc

    except Exception as e:
        print(f"[Trial {trial.number}] 실패: {str(e)}")
        return float("inf")
