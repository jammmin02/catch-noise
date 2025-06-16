import sounddevice as sd
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import time

import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우 한글 폰트 설정
matplotlib.rcParams['axes.unicode_minus'] = False     # 음수 깨짐 방지

# 설정
sr = 22050
segment_duration = 2.0
n_mfcc = 13
hop_length = 512
frame_per_second = sr / hop_length
max_len = int(frame_per_second * segment_duration)
class_names = ['non_noisy', 'noisy']
model_path = "../model/cnn_lstm_model.h5"

# 모델 로드
model = load_model(model_path)

# 특징 추출 함수
def extract_features(y_audio):
    mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(y=y_audio, hop_length=hop_length)
    features = np.vstack([mfcc, zcr])
    if features.shape[1] < max_len:
        features = np.pad(features, ((0, 0), (0, max_len - features.shape[1])), mode='constant')
    else:
        features = features[:, :max_len]
    return features.T[np.newaxis, ..., np.newaxis]

# 실시간 예측
print("실시간 예측 시작 (Ctrl+C로 종료)")
try:
    while True:
        audio = sd.rec(int(segment_duration * sr), samplerate=sr, channels=1, dtype='float32')
        sd.wait()
        audio = audio.flatten()

        x = extract_features(audio)
        pred = model.predict(x, verbose=0)[0]
        label_idx = int(pred[0] > 0.5)
        confidence = pred[0] if label_idx == 1 else 1 - pred[0]
        label = class_names[label_idx]

        # 시각화
        plt.clf()
        plt.subplot(2, 1, 1)
        librosa.display.waveshow(audio, sr=sr)
        plt.title("입력 음성 파형")

        plt.subplot(2, 1, 2)
        bars = plt.bar(class_names, [1 - pred[0], pred[0]], color=['blue', 'red'])
        bars[label_idx].set_color('green')
        plt.ylim([0, 1])
        plt.title(f"예측 결과: {label} ({confidence:.2f})")

        plt.pause(0.1)

except KeyboardInterrupt:
    print("\n실시간 예측 종료됨.")
    plt.close()
