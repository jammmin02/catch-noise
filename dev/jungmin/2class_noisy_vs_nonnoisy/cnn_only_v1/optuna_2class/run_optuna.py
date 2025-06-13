import os
import json
import optuna
import mlflow
import torch
from datetime import datetime
from model import CNNOnly
from objective_fn import objective
from data_loader import load_data
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

torch.backends.cudnn.enabled = False

# ✅ MLflow 설정
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://210.101.236.174:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ✅ 실험 이름
now = datetime.now()
EXPERIMENT_NAME = f"optuna_cnn_2class_{now.strftime('%Y%m%d_%H%M%S')}"
mlflow.set_experiment(EXPERIMENT_NAME)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_TRIALS = 20
    trial_counter = {"current": 0}

    # ✅ Optuna objective 함수 래핑
    def obj(trial):
        trial_counter["current"] += 1
        print(f"\n🧪 Trial {trial_counter['current']}/{N_TRIALS} 시작 중... (Optuna Trial #{trial.number})")
        try:
            return objective(trial, device)
        except Exception as e:
            print(f"⚠️ Trial {trial.number} 실패: {str(e)}")
            return float("inf")

    # ✅ Optuna 실행
    print(f"📊 총 {N_TRIALS}개의 Trial을 실행합니다...\n")
    study = optuna.create_study(direction="minimize")
    study.optimize(obj, n_trials=N_TRIALS)

    # ✅ Best Trial 결과
    best = study.best_trial
    best_val_acc = 1.0 - best.value
    print(f"\n🎉 ✅ Best Trial {best.number} 완료!")
    print(f"🏆 Validation Accuracy: {best_val_acc:.4f}")
    print(f"📌 Best Params:\n{json.dumps(best.params, indent=2)}")

    # ✅ MLflow 기록
    with mlflow.start_run(run_name=f"best_trial_{best.number}"):
        for k, v in best.params.items():
            mlflow.log_param(k, v)
        mlflow.log_metric("best_val_accuracy", best_val_acc)

        os.makedirs("outputs/cnn_only", exist_ok=True)
        with open("outputs/cnn_only/best_params.json", "w") as f:
            json.dump(best.params, f)
        mlflow.log_artifact("outputs/cnn_only/best_params.json")

        # ✅ 모델 정의 및 학습 준비
        model = CNNOnly(
            conv1_filters=best.params["conv1_filters"],
            conv2_filters=best.params["conv2_filters"],
            dense_units=best.params["dense_units"],
            dropout=best.params["dropout"]
        ).to(device)

        train_loader, val_loader, _ = load_data(best.params["batch_size"])
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=best.params["lr"])

        # ✅ 재학습
        print("\n📦 Best 모델 학습 시작 (재학습)...")
        for epoch in range(1, 11):
            model.train()
            running_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                yb = yb.view(-1, 1)

                preds = model(xb)
                loss = loss_fn(preds, yb)
                optimizer.zero_grad()
                optimizer.step()
                running_loss += loss.item()
            avg_loss = running_loss / len(train_loader)
            print(f"📘 Epoch {epoch:2d}/10 - 평균 Loss: {avg_loss:.4f}")

        # ✅ 모델 저장
        model_path = "outputs/cnn_only/best_model.pt"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)
        print(f"\n✅ Best model 저장 및 로그 완료 → {model_path}")

        # ✅ 검증 평가 및 시각화
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                yb = yb.view(-1, 1)

                output = model(xb).cpu().squeeze().numpy()
                target_np = yb.cpu().squeeze().numpy()
                pred_np = (output > 0.5).astype(int).tolist()
                target_np = target_np.astype(int).tolist()

                preds.extend(pred_np)
                targets.extend(target_np)

        acc = accuracy_score(targets, preds)
        f1 = f1_score(targets, preds)
        cm = confusion_matrix(targets, preds)

        mlflow.log_metric("final_val_accuracy", acc)
        mlflow.log_metric("final_val_f1_score", f1)

        # 🎨 혼동행렬 시각화
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")

        # 🎨 F1 Score 그래프
        plt.figure()
        plt.bar(["F1 Score"], [f1])
        plt.ylim(0, 1)
        plt.title("F1 Score")
        plt.savefig("f1_score.png")
        mlflow.log_artifact("f1_score.png")

        # 📝 성능 리포트 저장
        with open("best_report.txt", "w") as f:
            f.write(f"📌 Best Trial #{best.number}\n")
            f.write(f"Validation Accuracy: {acc:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(classification_report(targets, preds, digits=4))
        mlflow.log_artifact("best_report.txt")

if __name__ == "__main__":
    main()
