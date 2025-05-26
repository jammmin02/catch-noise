import sys, os
sys.path.append(os.path.dirname(__file__))

import optuna
import mlflow
import json
import torch
import torch.nn as nn
from torch.optim import Adam
from data_loader import load_data
from objective_fn import objective, safe_log_mlflow
from model import CNNLSTM

# ✅ MLflow 설정
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://210.101.236.174:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("optuna_cnn_lstm_2class")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_batch = 32
    train_loader, val_loader, _, _, _ = load_data(dummy_batch)

    # ✅ objective 함수 감싸서 사용
    def obj(trial):
        bs = trial.suggest_categorical("batch_size", [16, 32, 64])
        train_loader_, val_loader_, _, _, _ = load_data(bs)
        return objective(trial, device, train_loader_, val_loader_)

    # ✅ Optuna 실행
    study = optuna.create_study(direction="minimize")
    study.optimize(obj, n_trials=20)

    # ✅ Best Trial 처리
    best = study.best_trial
    print(f"\n✅ Best Trial {best.number}: val_acc={1.0 - best.value:.4f}")

    with mlflow.start_run(run_name=f"best_trial_{best.number}"):
        # 1. 로그
        for k, v in best.params.items():
            mlflow.log_param(k, v)
        mlflow.log_metric("best_val_accuracy", 1.0 - best.value)

        # 2. 파라미터 저장
        with open("best_params.json", "w") as f:
            json.dump(best.params, f)
        mlflow.log_artifact("best_params.json")

        # 3. 모델 재학습
        bs = best.params["batch_size"]
        train_loader, _, _, _, _ = load_data(bs)

        model = CNNLSTM(
            conv1_filters=best.params["conv1_filters"],
            conv2_filters=best.params["conv2_filters"],
            lstm_units=best.params["lstm_units"],
            dense_units=best.params["dense_units"],
            dropout=best.params["dropout"]
        ).to(device)

        loss_fn = nn.BCELoss()
        optimizer = Adam(model.parameters(), lr=best.params["lr"])

        for epoch in range(10):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = loss_fn(preds, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 4. 모델 저장
        model_path = "best_model.pt"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)

        print("✅ Best model 저장 및 로그 완료")

if __name__ == "__main__":
    main()
