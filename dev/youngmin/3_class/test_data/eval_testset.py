import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model

# ✅ 파일 경로 설정
X_PATH = 'dev/youngmin/test_outputs/X_test.npy'
Y_PATH = 'dev/youngmin/test_outputs/y_test.npy'
MODEL_PATH = 'dev/youngmin/outputs/cnn_lstm_model_3class.h5'

# ✅ 데이터 로딩
X_test = np.load(X_PATH)
y_true = np.load(Y_PATH)

# ✅ CNN 입력 형태로 reshape
X_test = X_test[..., np.newaxis]

# ✅ 모델 로딩
model = load_model(MODEL_PATH)

# ✅ 예측
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# ✅ 혼동 행렬 시각화
labels = ['quiet', 'neutral', 'noisy']
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (External Test Set)")
plt.tight_layout()
plt.savefig("dev/youngmin/test_outputs/confusion_matrix_test.png")
plt.show()

# ✅ 분류 리포트 출력
print("🔍 Classification Report (Test Set):")
print(classification_report(y_true, y_pred, target_names=labels))
