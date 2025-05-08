import os
import numpy as np
from sklearn.model_selection import train_test_split
import mlflow

def load_data():
    """
    📥 전처리된 데이터를 불러오고 train/val/test로 분할한 뒤 반환합니다.
    또한 timesteps, features도 함께 계산하여 반환합니다.
    """
    base_dir = os.path.join(os.getcwd(), "..", "outputs", "cnn_lstm")
    X_path = os.path.join(base_dir, "X_lstm.npy")
    y_path = os.path.join(base_dir, "y_lstm.npy")

    # 데이터 불러오기
    X = np.load(X_path)[..., np.newaxis]  # 채널 차원 추가
    y = np.load(y_path)

    # 데이터 분할
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=2/9, stratify=y_temp, random_state=42)

    # 입력 차원 계산 (Conv2D → Reshape용)
    timesteps = X.shape[1] // 4
    features = (X.shape[2] // 4) * 64

    # 📈 MLflow에 데이터 정보 기록
    mlflow.log_param("X_shape", str(X.shape))
    mlflow.log_param("y_distribution", str(np.bincount(y)))
    mlflow.log_param("X_train_shape", str(X_train.shape))
    mlflow.log_param("X_val_shape", str(X_val.shape))
    mlflow.log_param("X_test_shape", str(X_test.shape))
    mlflow.log_param("timesteps", timesteps)
    mlflow.log_param("features", features)

    return X_train, X_val, X_test, y_train, y_val, y_test, timesteps, features