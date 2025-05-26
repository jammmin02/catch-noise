import tf2onnx
from tensorflow import keras
import tensorflow as tf

# 모델 로드
model = keras.models.load_model("hyochan/jetson/dataset/outputs/cnn_lstm/cnn_lstm_model.keras")

# Dummy 입력 설정
input_shape = model.input_shape  # 예: (None, 86, 14, 1)
spec = (tf.TensorSpec(input_shape, tf.float32, name="input"),)

# ONNX 모델로 변환
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

# 저장
with open("hyochan/cnn_lstm_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
