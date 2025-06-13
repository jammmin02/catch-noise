import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# MLflow 서버 URI 설정
MLFLOW_TRACKING_URI = "http://210.101.236.174:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# robust_v7 기준 outputs 경로 통일
base_dir = "/app/dev/jungmin/2class_noisy_vs_nonnoisy/cnn_only_v1/outputs"
output_dir = os.path.join(base_dir, "analysis_outputs")
os.makedirs(output_dir, exist_ok=True)

# 분석할 최신 실험 이름 (자동으로 가장 최근 optuna 실험 찾기)
client = mlflow.tracking.MlflowClient()
experiments = client.search_experiments(filter_string="name LIKE 'optuna_cnn_2class_%'", order_by=["last_update_time DESC"])

if not experiments:
    print("분석 가능한 optuna_cnn_2class 실험이 없습니다.")
    exit()

# 가장 최근 실험 선택
experiment = experiments[0]
EXPERIMENT_NAME = experiment.name
experiment_id = experiment.experiment_id

print(f"Analyzing Experiment: {EXPERIMENT_NAME} (ID: {experiment_id})")

# 모든 run 조회
try:
    df = mlflow.search_runs(experiment_ids=[experiment_id])
except Exception as e:
    print(f"Run 조회 실패: {e}")
    exit()

if df.empty:
    print("No runs found for this experiment.")
    exit()

# Run 이름 정리
df["run_name"] = df.get("tags.mlflow.runName", df["run_id"].str.slice(0, 8))

# 분석에 필요한 컬럼만 필터링
columns_needed = [
    "run_name", "metrics.best_val_accuracy",
    "params.conv1_filters", "params.conv2_filters",
    "params.dense_units", "params.dropout", "params.lr", "params.batch_size"
]
existing_columns = [col for col in columns_needed if col in df.columns]
df_filtered = df[existing_columns].copy()

# 컬럼 이름 간소화
df_filtered.rename(columns={
    "metrics.best_val_accuracy": "val_accuracy",
    "params.batch_size": "batch",
    "params.conv1_filters": "conv1",
    "params.conv2_filters": "conv2",
    "params.dense_units": "dense",
    "params.dropout": "dropout",
    "params.lr": "lr"
}, inplace=True)

# 정확도 내림차순 정렬
df_sorted = df_filtered.sort_values("val_accuracy", ascending=False).reset_index(drop=True)

# 상위 5개 출력
print("\nTop 5 Trials:")
print(df_sorted[["run_name", "val_accuracy", "conv1", "conv2", "dense", "lr"]].head(5))

# 정확도 상위 5개 막대그래프
plt.figure(figsize=(10, 6))
sns.barplot(x="run_name", y="val_accuracy", data=df_sorted.head(5), palette="viridis")
plt.title("Top 5 Trials by Validation Accuracy")
plt.ylabel("Validation Accuracy")
plt.xlabel("Run Name")
plt.ylim(0, 1)
plt.xticks(rotation=30)
plt.tight_layout()

plot_path = os.path.join(output_dir, "top5_accuracy.png")
plt.savefig(plot_path)
plt.close()
print("시각화 저장 완료:", plot_path)

# 전체 결과 CSV 저장
csv_path = os.path.join(output_dir, "trial_summary.csv")
df_sorted.to_csv(csv_path, index=False)
print("CSV 저장 완료:", csv_path)
print("분석 완료!")
