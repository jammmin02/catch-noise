from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ✅ 테스트셋 다시 로딩
X_test = np.load('dev/youngmin/outputs/X_lstm.npy')
y_true = np.load('dev/youngmin/outputs/y_lstm.npy')

# ✅ reshape 및 모델 로딩
X_test = X_test[..., np.newaxis]
from tensorflow.keras.models import load_model
model = load_model('dev/youngmin/outputs/cnn_lstm_model_3class.h5')

# ✅ 예측
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# ✅ 혼동 행렬
cm = confusion_matrix(y_true, y_pred)
labels = ['quiet', 'neutral', 'noisy']

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (3-Class)")
plt.savefig("dev/youngmin/outputs/confusion_matrix.png")
plt.show()

# ✅ 정밀도/재현율 보고서
print("🔍 Classification Report:")
print(classification_report(y_true, y_pred, target_names=labels))
