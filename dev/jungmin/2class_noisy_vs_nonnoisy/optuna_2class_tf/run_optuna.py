import optuna
import mlflow
import os
from data_loader import load_data
from objective_fn import objective

# 🔧 실험 환경 설정
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://210.101.236.174:5000")
EXPERIMENT_NAME = "optuna_cnn_lstm_2class"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# 🎯 Optuna 튜닝 실행
def main():
    # 1. 데이터 로드
    with mlflow.start_run(run_name="optuna_tuning_session"):
        X_train, X_val, X_test, y_train, y_val, y_test, timesteps, features = load_data()

        # 2. study 생성
        study = optuna.create_study(direction="minimize")  # 또는 maximize

        # 3. optimize 실행 (필요 시 n_trials 조절 가능)
        study.optimize(
            lambda trial: objective(trial, X_train, X_val, y_train, y_val, timesteps, features),
            n_trials=20
        )

        # 4. 최적 파라미터 기록
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_val_loss", study.best_value)

        # 5. 저장 (JSON 등으로 따로 저장하고 싶으면 여기에 추가 가능)
        print("🎯 Best params:", study.best_params)
        print("📉 Best val loss:", study.best_value)

if __name__ == "__main__":
    main()
