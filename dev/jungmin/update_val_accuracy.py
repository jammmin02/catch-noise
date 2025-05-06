import mlflow

run_id = "22199513e94841bcb908facd9806fa97"  # ⬅️ 정민이의 model_2 Run ID

mlflow.set_experiment("Recovered_Experiments")

with mlflow.start_run(run_id=run_id):
    with open("dev/jungmin/3_class_modify/model_2/model_summary.txt", "r") as f:
        for line in f:
            if "val_accuracy" in line:
                val = float(line.strip().split(":")[-1])
                mlflow.log_metric("val_accuracy", val)
                print(f"✅ val_accuracy {val} 기록 완료!")
