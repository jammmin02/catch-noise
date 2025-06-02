import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# MLflow 서버 URI 설정
MLFLOW_TRACKING_URI = "http://210.101.236.174:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# 분석할 실험 이름 (날짜별 실험에 맞게 수정 필요)
EXPERIMENT_NAME = "optuna_cnn_2class_20250527_XXXXXX"
print(f"Analyzing Experiment: {EXPERIMENT_NAME}")

def main():
    # MLflow Experiment 검색
    try:
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    except Exception as e:
        print(f"MLflow 연결 실패: {e}")
        return

    if experiment is None:
        print(f"Experiment '{EXPERIMENT_NAME}' not found!")
        return

    experiment_id = experiment.experiment_id
    print(f"Experiment ID: {experiment_id}")

    # 해당 실험의 모든 run 조회
    try:
        df = mlflow.search_runs(experiment_ids=[experiment_id])
    except Exception as e:
        print(f"Run 조회 실패: {e}")
        return

    if df.empty:
        print("No runs found for this experiment.")
        return

    # Run 이름 설정 (run_name 컬럼 보정)
    df["run_name"] = df.get("tags.mlflow.runName", df["run_id"].str.slice(0, 8))

    # 분석에 필요한 주요 컬럼만 필터링
    columns_needed = [
        "run_name", "metrics.val_accuracy",
        "params.conv1_filters", "params.conv2_filters",
        "params.dense_units", "params.dropout", "params.lr", "params.batch_size"
    ]
    existing_columns = [col for col in columns_needed if col in df.columns]
    df_filtered = df[existing_columns].copy()

    # 컬럼 이름 간소화
    df_filtered.rename(columns={
        "metrics.val_accuracy": "val_accuracy",
        "params.batch_size": "batch",
        "params.conv1_filters": "conv1",
        "params.conv2_filters": "conv2",
        "params.dense_units": "dense",
        "params.dropout": "dropout",
        "params.lr": "lr"
    }, inplace=True)

    # val_accuracy 기준 내림차순 정렬
    df_sorted = df_filtered.sort_values("val_accuracy", ascending=False).reset_index(drop=True)

    # 상위 5개 출력
    print("\nTop 5 Trials:")
    print(df_sorted[["run_name", "val_accuracy", "conv1", "conv2", "dense", "lr"]].head(5))

    # 분석 결과 저장 디렉토리 생성
    output_dir = "analysis_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # 상위 5개 정확도 시각화 (막대그래프)
    plt.figure(figsize=(10, 6))
    sns.barplot(x="run_name", y="val_accuracy", data=df_sorted.head(5), palette="viridis")
    plt.title("Top 5 Trials by Validation Accuracy")
    plt.ylabel("Validation Accuracy")
    plt.xlabel("Run Name")
    plt.ylim(0, 1)
    plt.xticks(rotation=30)
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "top5_accuracy.png"))
    plt.close()
    print("시각화 저장 완료:", os.path.join(output_dir, "top5_accuracy.png"))

    # 전체 결과 CSV 저장
    csv_path = os.path.join(output_dir, "trial_summary.csv")
    df_sorted.to_csv(csv_path, index=False)
    print("CSV 저장 완료:", csv_path)
    print("분석 완료!")

if __name__ == "__main__":
    main()
