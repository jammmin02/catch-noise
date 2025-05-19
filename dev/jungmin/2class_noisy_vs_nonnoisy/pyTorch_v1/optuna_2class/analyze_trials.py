import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 🧪 실험 이름
EXPERIMENT_NAME = "optuna_cnn_lstm_2class"

def main():
    # ✅ 실험 ID 가져오기
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        print(f"❌ Experiment '{EXPERIMENT_NAME}' not found!")
        return
    experiment_id = experiment.experiment_id

    # ✅ 모든 Run 데이터 불러오기
    print("📥 Loading MLflow runs...")
    df = mlflow.search_runs(experiment_ids=[experiment_id])
    if df.empty:
        print("❌ No runs found.")
        return

    # ✅ run_name이 tags에 있는 경우 추출
    if "tags.mlflow.runName" in df.columns:
        df["run_name"] = df["tags.mlflow.runName"]
    else:
        df["run_name"] = df["run_id"].str.slice(0, 8)  # 없으면 run_id 일부 사용

    # ✅ 필요한 열만 추출 및 정리
    columns_needed = [
        "run_id", "run_name",
        "metrics.val_accuracy",
        "params.batch_size",
        "params.conv1_filters",
        "params.conv2_filters",
        "params.lstm_units",
        "params.dense_units",
        "params.dropout",
        "params.lr"
    ]

    # 존재하는 열만 필터링
    existing_columns = [col for col in columns_needed if col in df.columns]
    df_filtered = df[existing_columns].copy()

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

    # 🔢 정렬
    df_sorted = df_filtered.sort_values("val_accuracy", ascending=False).reset_index(drop=True)

    # ✅ 콘솔에 상위 5개 요약
    display_columns = [col for col in ["run_name", "val_accuracy", "conv1", "lstm", "lr"] if col in df_sorted.columns]
    print("\n🔝 Top 5 Trials (val_accuracy 기준):")
    print(df_sorted[display_columns].head(5))

    # ✅ 시각화 저장
    os.makedirs("analysis_outputs", exist_ok=True)
    top5 = df_sorted.head(5)

    if "run_name" in top5.columns and "val_accuracy" in top5.columns:
        plt.figure(figsize=(10, 6))
        sns.barplot(x="run_name", y="val_accuracy", data=top5, palette="viridis")
        plt.title("Top 5 Trials by Validation Accuracy")
        plt.ylabel("Validation Accuracy")
        plt.xlabel("Run Name")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig("analysis_outputs/top5_accuracy.png")
        plt.close()
        print("📊 Saved: analysis_outputs/top5_accuracy.png")
    else:
        print("⚠️ 시각화에 필요한 컬럼이 부족해서 그래프를 생략합니다.")

    # ✅ CSV 저장
    csv_path = "analysis_outputs/trial_summary.csv"
    df_sorted.to_csv(csv_path, index=False)
    print(f"📄 Saved: {csv_path}")

    print("✅ 분석 완료!")

if __name__ == "__main__":
    main()
