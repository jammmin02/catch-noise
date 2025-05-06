import mlflow
import os
from pathlib import Path

# Experiment 생성 또는 가져오기
mlflow.set_tracking_uri("file:///app/mlruns")  # Docker 기준 경로
mlflow.set_experiment("Recovered_Experiments")

# 등록할 실험 폴더들
experiment_dirs = {
    "outputs_test_1": "dev/jungmin/3_class/dataset/outputs_test_1/cnn_lstm",
    "outputs_test_2": "dev/jungmin/3_class/dataset/outputs_test_2/cnn_lstm",
    "outputs_test_3": "dev/jungmin/3_class/dataset/outputs_test_3/cnn_lstm",
    "model_1": "dev/jungmin/3_class_modify/model_1",
    "model_2": "dev/jungmin/3_class_modify/model_2",
    "model": "dev/jungmin/3_class_modify/model"
}

# 등록할 대표 파일 이름
loggable_files = [
    "cnn_lstm_model.h5",
    "train_history_3class_segment3s.png",
    "confusion_matrix_segment3s.png",
    "softmax_confidence_hist.png",
    "cnn_lstm_3class_performance_summary_v1.png",
    "model_summary.txt"
]

for run_name, path in experiment_dirs.items():
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("segment_duration", 3.0)
        mlflow.log_param("architecture", "CNN2D + LSTM")
        mlflow.log_metric("val_accuracy", 0.0)  # 정확도는 수동 입력 or unknown 처리

        for fname in loggable_files:
            fpath = os.path.join(path, fname)
            if os.path.exists(fpath):
                mlflow.log_artifact(fpath)

print("✅ 모든 과거 실험이 MLflow에 등록되었습니다.")
