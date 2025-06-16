import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Reshape, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import mlflow
import mlflow.tensorflow

# ========== MLflow 실험 시작 ==========
mlflow.set_experiment("NoiseClassification-CNN-LSTM")
with mlflow.start_run():

    # 1. 데이터 로딩
    x = np.load("data/processed/x.npy")
    y = np.load("data/processed/y.npy")
    x = x[..., np.newaxis]

    # 2. 훈련/검증 분할
    split = int(0.8 * len(x))
    x_train, x_val = x[:split], x[split:]
    y_train, y_val = y[:split], y[split:]

    # 3. 모델 구성
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(86, 14, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Reshape((21, 64 * 3)),
        LSTM(64),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # 4. 저장 경로
    os.makedirs("outputs", exist_ok=True)
    model_path = "outputs/cnn_lstm_model.h5"
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint(model_path, save_best_only=True)
    ]

    # 5. 하이퍼파라미터 로깅
    mlflow.log_params({
        "batch_size": 32,
        "epochs": 30,
        "learning_rate": 0.001,
        "input_shape": str((86, 14, 1)),
        "optimizer": "Adam",
        "model_type": "CNN + LSTM"
    })

    # 6. 학습
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # 7. 평가
    pred = model.predict(x_val)
    pred_label = (pred > 0.5).astype("float32")
    acc = accuracy_score(y_val, pred_label)
    cm = confusion_matrix(y_val, pred_label)

    mlflow.log_metric("val_accuracy", acc)

    # 8. 혼동 행렬 시각화
    cm_path = "outputs/confusion_matrix_h5.png"
    plt.figure(figsize=(5, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (.h5 Model)")
    plt.colorbar()

    classes = ['non_noisy', 'noisy']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=14)

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)

    # 9. 학습 그래프 시각화
    graph_path = "outputs/train_val_graphs.png"
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label="Train")
    plt.plot(history.history['val_accuracy'], label="Val")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label="Train")
    plt.plot(history.history['val_loss'], label="Val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(graph_path)
    mlflow.log_artifact(graph_path)

    # 10. 모델 저장 및 로깅
    model.save(model_path)
    mlflow.log_artifact(model_path)

    print("MLflow 실험 기록 완료")
