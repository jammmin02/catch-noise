import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Reshape, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 🔧 경로 설정
base_dir = "hyochan/model_make_test/dataset/outputs/cnn_lstm"
X_path = os.path.join(base_dir, "X_lstm.npy")
y_path = os.path.join(base_dir, "y_lstm.npy")
model_save_path = os.path.join(base_dir, "cnn_lstm_model.h5")
plot_save_path = os.path.join(base_dir, "train_history_3class_segment3s.png")

# 📥 데이터 로드
X = np.load(X_path)  # (샘플 수, max_len, 14)
y = np.load(y_path)
print(f"✅ Data loaded: X shape = {X.shape}, y shape = {y.shape}")
print(f"🧾 Label distribution: {np.bincount(y)}")  # 클래스 분포 확인

# 📐 CNN 입력 형태로 reshape
X = X[..., np.newaxis]  # (샘플 수, max_len, 14, 1)

# 📊 데이터셋 분할 (7:2:1)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=2/9, stratify=y_temp, random_state=42
)

print(f"📊 Split sizes → Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

# 🧠 모델 정의 (3-class + softmax)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(X.shape[1], X.shape[2], 1)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),

    Reshape((X.shape[1] // 4, -1)),  # 두 번 MaxPooling → 시간축 1/4
    LSTM(64),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')  # ✅ 3 클래스!
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 🏁 학습
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# 🧪 테스트셋 평가
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"🧪 Test Accuracy: {test_acc:.4f}, Loss: {test_loss:.4f}")

# 💾 모델 저장
model.save(model_save_path)
print(f"✅ Model saved: {model_save_path}")

# 📈 학습 그래프 저장 및 출력
plt.figure(figsize=(12, 5))

# Accuracy 그래프
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc', marker='o')
plt.plot(history.history['val_accuracy'], label='Val Acc', marker='x')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss 그래프
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Val Loss', marker='x')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(plot_save_path)
plt.show()
print(f"📈 Plot saved: {plot_save_path}")
