import mlflow
import os

# 🔧 경로 설정
local_artifact_dir = "hyochan/jetson/dataset/outputs/cnn_lstm"
mlflow.set_tracking_uri("http://210.101.236.174:5000")
mlflow.set_experiment("train_cnn_lstm_2class_hyochan_20250523_032401")

# 새로 업로드할 파일 목록
upload_files = [
    "cnn_lstm_model.keras",
    "train_history.png",
    "confusion_matrix.png",
    "confidence_hist.png",
    "model_summary.txt",
    "X_lstm.npy",
    "y_lstm.npy",
]

with mlflow.start_run(run_id="e5d633485f484a87bdad6194386039bd"):
    for filename in upload_files:
        file_path = os.path.join(local_artifact_dir, filename)
        mlflow.log_artifact(file_path, artifact_path="cnn_lstm")  # 📦 artifact 위치 유지

print("✅ MLflow artifact 업로드 완료!")
