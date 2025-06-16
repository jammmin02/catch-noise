import mlflow
import os

# 로컬 artifact 경로
local_artifact_dir = "hyochan/pc/dataset/outputs/cnn_lstm"

# MLflow 서버 주소 및 실험 이름
mlflow.set_tracking_uri("http://210.101.236.174:5000")
mlflow.set_experiment("train_cnn_lstm_2class_hyochan_20250528_132555")

# 업로드할 파일 목록
upload_files = [
    "cnn_lstm_model.keras",
    "train_history.png",
    "confusion_matrix.png",
    "confidence_hist.png",
    "model_summary.txt",
    "X_lstm.npy",
    "y_lstm.npy",
]

# 업로드할 Run ID (UI에서 정확히 복사한 값)
run_id = "99ab09936d7b4a4fbe8a6e9e8c82124d"

# ✅ Run 시작 (기존 run_id에 연결)
with mlflow.start_run(run_id=run_id):
    print(f"MLflow Artifact 업로드 시작 (Run ID: {run_id})")

    for filename in upload_files:
        file_path = os.path.join(local_artifact_dir, filename)

        if os.path.exists(file_path):
            mlflow.log_artifact(file_path, artifact_path="cnn_lstm")  # 원하는 폴더에 저장
            print(f"업로드 완료: {filename}")
        else:
            print(f"파일 없음: {file_path}")

print("모든 artifact 업로드 완료!")
