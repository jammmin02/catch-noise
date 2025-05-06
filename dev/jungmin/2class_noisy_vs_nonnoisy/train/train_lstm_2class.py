import os
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Reshape, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import BinaryCrossentropy

# ✅ 날짜 기반 실험 이름
today = datetime.now().strftime("%Y%m%d")
experiment_name = f"train_cnn_lstm_2class_jungmin_{today}"

# ✅ MLflow 설정
mlflow.set_tracking_uri("file:/app/mlruns")
mlflow.set_experiment(experiment_name)

with mlflow.start_run():
    # 🔧 기본 설정
    segment_duration = 3.0
    base_dir = "dev/jungmin/2class_noisy_vs_nonnoisy/outputs/cnn_lstm"
    X_path = os.path.join(base_dir, "X_lstm.npy")
    y_path = os.path.join(base_dir, "y_lstm.npy")
    model_save_path = os.path.join(base_dir, "cnn_lstm_model.h5")
    plot_save_path = os.path.join(base_dir, f"train_history_segment{int(segment_duration)}s.png")
    confusion_path = os.path.join(base_dir, "confusion_matrix_segment3s.png")
    confidence_plot_path = os.path.join(base_dir, "sigmoid_confidence_hist.png")
    model_summary_path = os.path.join(base_dir, "model_summary.txt")
    label_names = ['non_noisy', 'noisy']

    # 📥 데이터 로드
    X = np.load(X_path)
    y = np.load(y_path)
    X = X[..., np.newaxis]

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=2/9, stratify=y_temp, random_state=42)

    # 🧠 모델 정의
    timesteps = X.shape[1] // 4
    features = (X.shape[2] // 4) * 64

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(X.shape[1], X.shape[2], 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Reshape((timesteps, features)),
        LSTM(64),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss=BinaryCrossentropy(), metrics=['accuracy'])

    # ✅ 모델 요약 저장 및 로깅
    with open(model_summary_path, "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))
    mlflow.log_artifact(model_summary_path)

    # ✅ 파라미터 로깅
    mlflow.log_params({
        "segment_duration": segment_duration,
        "optimizer": "adam",
        "loss": "binary_crossentropy",
        "batch_size": 32,
        "epochs": 30,
        "architecture": "cnn_lstm"
    })

    # 🔁 학습
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    # 🧪 평가
    test_loss, test_acc = model.evaluate(X_test, y_test)
    mlflow.log_metrics({
        "test_accuracy": test_acc,
        "test_loss": test_loss
    })

    # ✅ 모델 저장 및 로깅
    model.save(model_save_path)
    mlflow.log_artifact(model_save_path)

    # 📈 학습 그래프 시각화
    def smooth_curve(points, factor=0.6):
        smoothed = []
        for point in points:
            if smoothed:
                smoothed.append(smoothed[-1] * factor + point * (1 - factor))
            else:
                smoothed.append(point)
        return smoothed

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc', marker='o')
    plt.plot(history.history['val_accuracy'], label='Val Acc', marker='x')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(smooth_curve(history.history['loss']), label='Train Loss (Smoothed)', marker='o')
    plt.plot(smooth_curve(history.history['val_loss']), label='Val Loss (Smoothed)', marker='x')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(plot_save_path)
    plt.show()
    mlflow.log_artifact(plot_save_path)

    # 📊 혼동 행렬
    y_pred = model.predict(X_test)
    y_pred_classes = (y_pred > 0.5).astype(int).reshape(-1)
    cm = confusion_matrix(y_test, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(confusion_path)
    plt.show()
    mlflow.log_artifact(confusion_path)

    # 🔍 Confidence 히스토그램
    confidences = y_pred.flatten()
    plt.figure(figsize=(8, 5))
    plt.hist(confidences, bins=20, color='skyblue', edgecolor='black')
    plt.title("Sigmoid Confidence Distribution (Test Set)")
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(confidence_plot_path)
    plt.show()
    mlflow.log_artifact(confidence_plot_path)

print(f"✅ MLflow experiment '{experiment_name}' 저장 완료!")
