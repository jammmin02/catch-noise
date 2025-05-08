import os
import json
import numpy as np
import mlflow
import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Reshape, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 📁 데이터 경로
project_root = os.getcwd()
data_dir = os.path.join(project_root, "..", "outputs", "cnn_lstm")
X = np.load(os.path.join(data_dir, "X_lstm.npy"))[..., np.newaxis]
y = np.load(os.path.join(data_dir, "y_lstm.npy"))

# ✅ 전체 데이터로 다시 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
timesteps = X.shape[1] // 4
features = (X.shape[2] // 4) * 64

# ✅ best_params 수동 입력 or 불러오기
# 아래는 예시 — 실제 튜닝 후 study.best_params 복사해서 넣자
best_params = {
    "conv1_filters": 32,
    "conv2_filters": 64,
    "lstm_units": 64,
    "dense_units": 64,
    "dropout": 0.3,
    "lr": 0.001,
    "batch_size": 32
}

# ✅ MLflow 설정
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://210.101.236.174:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("optuna_cnn_lstm_2class")

with mlflow.start_run(run_name="train_best_model"):
    # 모델 구성
    model = Sequential()
    model.add(Conv2D(best_params["conv1_filters"], (3, 3), activation='relu', padding='same', input_shape=(X.shape[1], X.shape[2], 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(best_params["conv2_filters"], (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Reshape((timesteps, features)))
    model.add(LSTM(best_params["lstm_units"]))
    model.add(Dense(best_params["dense_units"], activation='relu'))
    model.add(Dropout(best_params["dropout"]))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_params["lr"]),
                  loss=BinaryCrossentropy(), metrics=["accuracy"])

    # 조기 종료
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # 학습
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=30,
        batch_size=best_params["batch_size"],
        callbacks=[early_stop],
        verbose=1
    )

    # 평가
    test_loss, test_acc = model.evaluate(X_test, y_test)
    mlflow.log_metrics({"test_loss": test_loss, "test_accuracy": test_acc})

    # 모델 저장
    model_path = os.path.join(data_dir, "final_best_model.keras")
    save_model(model, model_path)
    mlflow.log_artifact(model_path)

    # 📈 혼동 행렬
    y_pred = (model.predict(X_test) > 0.5).astype(int).reshape(-1)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["non_noisy", "noisy"])
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    cm_path = os.path.join(data_dir, "final_confusion_matrix.png")
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)
    plt.close()

    # 📊 학습 그래프
    plt.figure(figsize=(12, 4))
    plt.plot(history.history['accuracy'], label="Train Acc")
    plt.plot(history.history['val_accuracy'], label="Val Acc")
    plt.legend()
    plt.grid(True)
    plt.title("Accuracy Curve")
    acc_path = os.path.join(data_dir, "final_accuracy_curve.png")
    plt.savefig(acc_path)
    mlflow.log_artifact(acc_path)
    plt.close()

    # 🔢 하이퍼파라미터 전부 기록
    mlflow.log_params(best_params)

print("✅ 최종 모델 학습 및 기록 완료!")
