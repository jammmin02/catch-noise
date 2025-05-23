import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Reshape, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 🔧 경로 설정
base_dir = "outputs"
X_path = os.path.join(base_dir, "X_lstm_bview.npy")
y_path = os.path.join(base_dir, "y_lstm_bview.npy")
model_save_path = os.path.join(base_dir, "cnn_lstm_bview_model.h5")
plot_save_path = os.path.join(base_dir, "train_history_2class_bview_segment2s.png")

# 📥 데이터 불러오기
X = np.load(X_path)
y = np.load(y_path)
print(f"✅ 데이터 불러오기 완료: X shape = {X.shape}, y shape = {y.shape}")
print(f"🧾 라벨 분포: {np.bincount(y)}")

# 🔄 입력 형태 변환
X = X[..., np.newaxis]  # (샘플 수, 프레임 수, 특징 수, 채널 수)

# 📊 데이터셋 분할 (train:val:test = 7:2:1)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=2/9, stratify=y_temp, random_state=42)
print(f"📊 데이터 분할 완료 → Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

# 🧠 모델 구성 (CNN2D + LSTM)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(X.shape[1], X.shape[2], 1)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),

    Reshape((X.shape[1] // 4, -1)),
    LSTM(64),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# ⚙️ 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 🏁 학습 설정
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 🏋️ 모델 학습
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# 🧪 테스트 평가
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"🧪 테스트 정확도: {test_acc:.4f}, 손실: {test_loss:.4f}")

# 💾 모델 저장
model.save(model_save_path)
print(f"✅ 모델 저장 완료: {model_save_path}")

# 📈 학습 그래프 저장
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='x')
plt.title('정확도(Accuracy)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='x')
plt.title('손실(Loss)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(plot_save_path)
plt.show()
print(f"📈 학습 그래프 저장 완료: {plot_save_path}")
