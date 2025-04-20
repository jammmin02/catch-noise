import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Reshape, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# ✅ 경로 설정
DATA_DIR = 'dev/youngmin/outputs'
X_PATH = os.path.join(DATA_DIR, 'X_lstm.npy')
Y_PATH = os.path.join(DATA_DIR, 'y_lstm.npy')
MODEL_SAVE_PATH = os.path.join(DATA_DIR, 'cnn_lstm_model_3class.h5')

# ✅ 데이터 로딩
print(f"📂 Loading: {X_PATH}")
X = np.load(X_PATH)
print(f"📂 Loading: {Y_PATH}")
y = np.load(Y_PATH)

# ✅ 디버깅 정보
print(f"✅ X shape: {X.shape}")
print(f"✅ y shape: {y.shape}")
print(f"🧪 Unique labels in y: {np.unique(y)}")

# ✅ 비어 있는 y 예외 처리
if y.size == 0:
    raise ValueError("❗ y 라벨이 비어있습니다. 전처리 데이터를 확인하세요.")

# ✅ reshape: CNN 입력용으로 채널 추가
X = X[..., np.newaxis]

# ✅ 라벨 one-hot encoding
num_classes = len(np.unique(y))
y_cat = to_categorical(y, num_classes=num_classes)

# ✅ 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# ✅ 모델 구성
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=X.shape[1:]),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    # ⛳ 오류 해결 핵심: 자동 time step 계산
    Reshape((-1, 64)),

    LSTM(64),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

# ✅ 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ✅ 학습
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=32,
    callbacks=[early_stop]
)

# ✅ 평가
loss, acc = model.evaluate(X_test, y_test)
print(f"\n✅ 3-class 테스트 정확도: {acc * 100:.2f}%")

# ✅ 모델 저장
model.save(MODEL_SAVE_PATH)
print(f"📦 모델 저장 완료: {MODEL_SAVE_PATH}")

# ✅ 시각화 저장
plt.figure()
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy over Epochs')
plt.legend()
plt.savefig(os.path.join(DATA_DIR, 'cnn_lstm_train_accuracy.png'))

plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.savefig(os.path.join(DATA_DIR, 'cnn_lstm_train_loss.png'))

print("✅ 학습 그래프 저장 완료!")
