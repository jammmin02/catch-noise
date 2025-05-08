import sounddevice as sd
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import time

# 📌 설정
sr = 22050
n_mfcc = 13
hop_length = 512
segment_duration = 2.0
frame_per_second = sr / hop_length
max_len = int(frame_per_second * segment_duration)

# ✅ 2-class 설정
class_names = ['non_noisy', 'noisy']
model_path = "hyochan/model_make_test/dataset/outputs/cnn_lstm/cnn_lstm_model.h5"

# ✅ 모델 로딩
model = load_model(model_path)

# 🎛️ 전처리 함수
def extract_features(y_audio):
    mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(y=y_audio, hop_length=hop_length)
    features = np.vstack([mfcc, zcr])
    if features.shape[1] < max_len:
        features = np.pad(features, ((0, 0), (0, max_len - features.shape[1])), mode='constant')
    else:
        features = features[:, :max_len]
    return features.T[np.newaxis, ..., np.newaxis]  # (1, max_len, 14, 1)

# 🔁 실시간 루프
print("🎙️ 실시간 소음 예측 시작 (Ctrl+C로 종료)...")
try:
    while True:
        print("\n🎧 2초 동안 녹음 중...")
        audio = sd.rec(int(segment_duration * sr), samplerate=sr, channels=1, dtype='float32')
        sd.wait()
        audio = audio.flatten()

        x = extract_features(audio)
        pred = model.predict(x, verbose=0)[0]
        label_idx = np.argmax(pred)
        label = class_names[label_idx]
        confidence = pred[label_idx]

        print(f"🔊 예측 결과: [{label}] (신뢰도: {confidence:.2f})")
        time.sleep(0.5)

except KeyboardInterrupt:
    print("\n🛑 실시간 예측 종료됨.")
