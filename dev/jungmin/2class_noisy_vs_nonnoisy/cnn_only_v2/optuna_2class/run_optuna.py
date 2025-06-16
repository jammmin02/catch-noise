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

# ✅ 실험 이름 (v2로 명확하게 표시)
now = datetime.now()
EXPERIMENT_NAME = f"optuna_cnn_only_v2_{now.strftime('%Y%m%d_%H%M%S')}"
mlflow.set_experiment(EXPERIMENT_NAME)

# ✅ 경로 설정
BASE_DIR = "dev/jungmin/2class_noisy_vs_nonnoisy/cnn_only_v2"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

        # ✅ best_params.json 저장
        param_path = os.path.join(OUTPUT_DIR, "best_params.json")
        with open(param_path, "w") as f:
            json.dump(best.params, f)
        mlflow.log_artifact(param_path)

        # ✅ 모델 정의 및 학습 준비
        model = CNNOnly(
            conv1_filters=best.params["conv1_filters"],
            conv2_filters=best.params["conv2_filters"],
            dense_units=best.params["dense_units"],
            dropout=best.params["dropout"]
        ).to(device)

        train_loader, val_loader, _ = load_data(best.params["batch_size"])
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=best.params["lr"])

        # ✅ 재학습
        print("\n📦 Best 모델 학습 시작 (재학습)...")
        for epoch in range(1, 11):
            model.train()
            running_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = loss_fn(preds, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            avg_loss = running_loss / len(train_loader)
            print(f"📘 Epoch {epoch:2d}/10 - 평균 Loss: {avg_loss:.4f}")

        # ✅ Dummy forward 이후 저장 (TensorRT 대응)
        dummy_input = torch.randn(1, 1, 86, 13).to(device)  # <-- 전처리 input shape 기준
        _ = model(dummy_input)

        # ✅ 모델 저장
        model_path = os.path.join(OUTPUT_DIR, "best_model.pt")
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)
        print(f"\n✅ Best model 저장 및 로그 완료 → {model_path}")

        # ✅ 검증 평가 및 시각화
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb).cpu()
                pred_classes = torch.argmax(outputs, dim=1)
                preds.extend(pred_classes.numpy().tolist())
                targets.extend(yb.cpu().numpy().tolist())

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
        cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)

        # 🎨 F1 Score 그래프
        plt.figure()
        plt.bar(["F1 Score"], [f1])
        plt.ylim(0, 1)
        plt.title("F1 Score")
        f1_path = os.path.join(OUTPUT_DIR, "f1_score.png")
        plt.savefig(f1_path)
        mlflow.log_artifact(f1_path)

        # 📝 성능 리포트 저장
        report_path = os.path.join(OUTPUT_DIR, "best_report.txt")
        with open(report_path, "w") as f:
            f.write(f"📌 Best Trial #{best.number}\n")
            f.write(f"Validation Accuracy: {acc:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(classification_report(targets, preds, digits=4))
        mlflow.log_artifact(report_path)

if __name__ == "__main__":
    main()
