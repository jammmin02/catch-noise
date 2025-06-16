import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# ✅ 오늘 날짜 기준 실험 이름
EXPERIMENT_NAME = "optuna_cnn_lstm_2class_20250522_120900"
print(f"📁 Analyzing Experiment: {EXPERIMENT_NAME}")

def main():
    try:
        mlflow.set_tracking_uri("http://210.101.236.174:5000")
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    except Exception as e:
        print(f"❌ MLflow 연결 실패: {e}")
        return

    if experiment is None:
        print(f"❌ Experiment '{EXPERIMENT_NAME}' not found!")
        return

    experiment_id = experiment.experiment_id
    print(f"📥 Experiment ID: {experiment_id}")

    try:
        df = mlflow.search_runs(experiment_ids=[experiment_id])
    except Exception as e:
        print(f"❌ Run 조회 실패: {e}")
        return

    if df.empty:
        print("⚠️ No runs found for this experiment.")
        return

    # ✅ Run 이름 설정 (없을 경우 run_id 일부 사용)
    df["run_name"] = df.get("tags.mlflow.runName", df["run_id"].str.slice(0, 8))

    # ✅ 필요한 컬럼 필터링
    columns_needed = [
        "run_name", "metrics.val_accuracy",
        "params.conv1_filters", "params.conv2_filters",
        "params.lstm_units", "params.dense_units",
        "params.dropout", "params.lr", "params.batch_size"
    ]
    existing_columns = [col for col in columns_needed if col in df.columns]
    df_filtered = df[existing_columns].copy()

    # ✅ 컬럼 이름 단축
    df_filtered.rename(columns={
        "metrics.val_accuracy": "val_accuracy",
        "params.batch_size": "batch",
        "params.conv1_filters": "conv1",
        "params.conv2_filters": "conv2",
        "params.lstm_units": "lstm",
        "params.dense_units": "dense",
        "params.dropout": "dropout",
        "params.lr": "lr"
    }, inplace=True)

    # ✅ 성능 기준 정렬
    df_sorted = df_filtered.sort_values("val_accuracy", ascending=False).reset_index(drop=True)

    print("\n🔝 Top 5 Trials by val_accuracy:")
    print(df_sorted[["run_name", "val_accuracy", "conv1", "lstm", "lr"]].head(5))

    # ✅ 저장 폴더 생성
    os.makedirs("analysis_outputs", exist_ok=True)

    # ✅ Top 5 시각화 저장
    top5 = df_sorted.head(5)
    plt.figure(figsize=(10, 6))
    sns.barplot(x="run_name", y="val_accuracy", data=top5, palette="viridis")
    plt.title("Top 5 Trials by Validation Accuracy")
    plt.ylabel("Validation Accuracy")
    plt.xlabel("Run Name")
    plt.ylim(0, 1)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("analysis_outputs/top5_accuracy.png")
    plt.close()
    print("📊 저장 완료: analysis_outputs/top5_accuracy.png")

    # ✅ CSV 저장
    csv_path = "analysis_outputs/trial_summary.csv"
    df_sorted.to_csv(csv_path, index=False)
    print(f"📄 저장 완료: {csv_path}")
    print("✅ 분석 완료!")

if __name__ == "__main__":
    main()
