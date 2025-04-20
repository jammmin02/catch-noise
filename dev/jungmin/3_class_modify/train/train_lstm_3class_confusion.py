import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Reshape, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 🔧 설정
segment_duration = 3.0
base_dir = "dev/jungmin/3_class_modify/dataset/outputs/cnn_lstm"
X_path = os.path.join(base_dir, "X_lstm.npy")
y_path = os.path.join(base_dir, "y_lstm.npy")
model_save_path = os.path.join(base_dir, "cnn_lstm_model.h5")
plot_save_path = os.path.join(base_dir, f"train_history_3class_segment{int(segment_duration)}s.png")
model_summary_path = os.path.join(base_dir, "model_summary.txt")
confusion_path = os.path.join(base_dir, "confusion_matrix_segment3s.png")
confidence_plot_path = os.path.join(base_dir, "softmax_confidence_hist.png")

label_names = ['neutral', 'non_noisy', 'noisy']

# 📥 데이터 로드
X = np.load(X_path)
y = np.load(y_path)
print(f"✅ Data loaded: X shape = {X.shape}, y shape = {y.shape}")

# 라벨 분포 출력
for i, name in enumerate(label_names):
    print(f"   {name:>10}: {np.sum(y == i)}개")

X = X[..., np.newaxis]

# 데이터셋 분할
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=2/9, stratify=y_temp, random_state=42)
print(f"📊 Split sizes → Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

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
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 구조 저장
with open(model_summary_path, "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))

# 학습
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# 평가
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"🧪 Test Accuracy: {test_acc:.4f}, Loss: {test_loss:.4f}")

# 모델 저장
model.save(model_save_path)
print(f"✅ Model saved: {model_save_path}")

# 📈 학습 그래프 시각화
# Loss smoothing
def smooth_curve(points, factor=0.6):
    smoothed = []
    for point in points:
        if smoothed:
            smoothed.append(smoothed[-1] * factor + point * (1 - factor))
        else:
            smoothed.append(point)
    return smoothed

plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc', marker='o')
plt.plot(history.history['val_accuracy'], label='Val Acc', marker='x')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Smoothed Loss
plt.subplot(1, 2, 2)
train_loss_smoothed = smooth_curve(history.history['loss'])
val_loss_smoothed = smooth_curve(history.history['val_loss'])
plt.plot(train_loss_smoothed, label='Train Loss (Smoothed)', marker='o')
plt.plot(val_loss_smoothed, label='Val Loss (Smoothed)', marker='x')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(plot_save_path)
plt.show()
print(f"📈 Plot saved: {plot_save_path}")

# 📊 혼동 행렬 시각화
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_test, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(confusion_path)
plt.show()
print(f"📊 Confusion matrix saved: {confusion_path}")

# 🔍 Softmax confidence 시각화
confidences = np.max(y_pred, axis=1)
plt.figure(figsize=(8, 5))
plt.hist(confidences, bins=20, color='skyblue', edgecolor='black')
plt.title("Softmax Confidence Distribution (Test Set)")
plt.xlabel("Max Softmax Probability")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.savefig(confidence_plot_path)
plt.show()
print(f"📊 Softmax confidence histogram saved: {confidence_plot_path}")
