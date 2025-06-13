import os
import json
import optuna
import mlflow
import torch
import numpy as np
from datetime import datetime
from model import CNNOnly
from objective_fn import objective
from data_loader import load_data
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

torch.backends.cudnn.enabled = False

# MLflow 서버 주소
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://210.101.236.174:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# 실험명 자동 생성
now = datetime.now()
EXPERIMENT_NAME = f"optuna_cnn_2class_{now.strftime('%Y%m%d_%H%M%S')}"
mlflow.set_experiment(EXPERIMENT_NAME)

# robust_v7 기준 base_dir 통일
base_dir = "/app/dev/jungmin/2class_noisy_vs_nonnoisy/cnn_only_v1/outputs"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_TRIALS = 20
    trial_counter = {"current": 0}

    def obj(trial):
        trial_counter["current"] += 1
        print(f"\n[Trial {trial_counter['current']}/{N_TRIALS}] 시작 (Optuna Trial #{trial.number})")
        try:
            return objective(trial, device)
        except Exception as e:
            print(f"[Trial {trial.number}] 실패: {str(e)}")
            return float("inf")

    # Optuna 튜닝 시작
    print(f"\n총 {N_TRIALS}개의 하이퍼파라미터 탐색 시작\n")
    study = optuna.create_study(direction="minimize")
    study.optimize(obj, n_trials=N_TRIALS)

    # Best Trial 결과 출력
    best = study.best_trial
    best_val_acc = 1.0 - best.value
    print(f"\n[Best Trial #{best.number}] 완료")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print("Best Hyperparameters:")
    print(json.dumps(best.params, indent=2))

    # MLflow 기록 및 저장
    with mlflow.start_run(run_name=f"best_trial_{best.number}"):
        for k, v in best.params.items():
            mlflow.log_param(k, v)
        mlflow.log_metric("best_val_accuracy", best_val_acc)

        # robust_v7 기준 경로
        os.makedirs(base_dir, exist_ok=True)
        with open(os.path.join(base_dir, "best_params.json"), "w") as f:
            json.dump(best.params, f)
        mlflow.log_artifact(os.path.join(base_dir, "best_params.json"))

        # 최적 파라미터 재학습
        print("\n[최적 파라미터 재학습 시작]")

        # input_shape 자동 추출
        X_sample = np.load(os.path.join(base_dir, "X_cnn.npy"))
        input_shape = X_sample.shape[1:]  # (time_steps, n_features)

        model = CNNOnly(
            input_shape=input_shape,
            conv1_filters=best.params["conv1_filters"],
            conv2_filters=best.params["conv2_filters"],
            dense_units=best.params["dense_units"],
            dropout=best.params["dropout"]
        ).to(device)

        train_loader, val_loader, _ = load_data(best.params["batch_size"], base_dir=base_dir)

        loss_fn = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=best.params["lr"])

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

        # 모델 저장
        model_path = os.path.join(base_dir, "best_model.pt")
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)
        print(f"\n최종 모델 저장 완료 → {model_path}")

        # 검증 평가
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
        f1 = f1_score(targets, preds)
        cm = confusion_matrix(targets, preds)

        mlflow.log_metric("final_val_accuracy", acc)
        mlflow.log_metric("final_val_f1_score", f1)

        # 혼동행렬 시각화
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")

        # F1 Score 시각화
        plt.figure()
        plt.bar(["F1 Score"], [f1])
        plt.ylim(0, 1)
        plt.title("F1 Score")
        plt.savefig("f1_score.png")
        mlflow.log_artifact("f1_score.png")

        # 전체 classification report
        with open("best_report.txt", "w") as f:
            f.write(f"Best Trial #{best.number}\n")
            f.write(f"Validation Accuracy: {acc:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(classification_report(targets, preds, digits=4))
        mlflow.log_artifact("best_report.txt")

if __name__ == "__main__":
    main()
