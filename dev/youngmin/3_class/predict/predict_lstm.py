import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from tensorflow.keras.models import load_model

# ✅ 경로 설정
X_test_path = "dev/youngmin/dataset/outputs/X_test.npy"
y_test_path = "dev/youngmin/dataset/outputs/y_test.npy"
model_path = "dev/youngmin/results/model/cnn_lstm_model.h5"
plot_save_path = "dev/youngmin/results/plots/confusion_matrix.png"

# ✅ 데이터 로드
X_test = np.load(X_test_path)
y_test = np.load(y_test_path)
print(f"✅ 테스트셋 로드 완료: X={X_test.shape}, y={y_test.shape}")

# ✅ 모델 로드
model = load_model(model_path)
print("✅ 모델 로드 완료")

# ✅ 예측
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

# ✅ 정확도 출력
acc = accuracy_score(y_test, y_pred)
print(f"🎯 테스트 정확도: {acc * 100:.2f}%")

# ✅ 혼동 행렬 시각화
cm = confusion_matrix(y_test, y_pred)
labels = ['quiet', 'neutral', 'noisy']
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
plt.savefig(plot_save_path)
print(f"📊 혼동 행렬 저장 완료: {plot_save_path}")
plt.close()

# ✅ 테스트셋 저장
np.save("dev/youngmin/dataset/outputs/X_test.npy", X_test)
np.save("dev/youngmin/dataset/outputs/y_test.npy", y_test)
print("💾 테스트셋 저장 완료.")
