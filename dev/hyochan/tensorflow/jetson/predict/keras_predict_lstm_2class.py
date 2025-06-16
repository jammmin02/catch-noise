# -*- coding: utf-8 -*-

import sounddevice as sd
import librosa
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import threading
import time

# === 설정 ===
mic_sr = 44100                # 마이크 입력 샘플레이트
model_sr = 22050              # 모델 학습 시 사용된 샘플레이트
segment_duration = 2.0        # 입력 세그먼트 길이 (초)
n_mfcc = 13
hop_length = 512
max_len = 86                  # 모델 입력 프레임 수
class_names = ['non_noisy', 'noisy']
model_path = "hyochan/jetson_predict/cnn_lstm_model.keras"
model = load_model(model_path)

# === 마이크 장치 자동 탐색 ===
def find_input_device(name_keyword="usb"):
    for idx, dev in enumerate(sd.query_devices()):
        if name_keyword.lower() in dev["name"].lower() and dev["max_input_channels"] > 0:
            print(f"Selected microphone: {dev['name']} (index {idx})")
            return idx, dev["max_input_channels"]
    raise RuntimeError("USB microphone not found.")

device_index, device_channels = find_input_device("usb")

# === 실시간 변수 초기화 ===
rolling_audio = np.zeros(int(mic_sr * segment_duration), dtype=np.float32)
latest_pred = np.array([0.5, 0.5])
latest_label = "non_noisy"
latest_confidence = 0.5

# === 시각화 초기화 ===
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

bar_plot = ax1.bar(class_names, latest_pred, color=["skyblue", "salmon"])
ax1.set_ylim(0, 1)
ax1.set_title("Prediction results")
ax1.set_ylabel("Probability")

plot_len = int(mic_sr * 0.5)
line_wave, = ax2.plot(np.zeros(plot_len))
ax2.set_ylim(-1, 1)
ax2.set_title("Real-time microphone input")
ax2.set_xlabel("Sample")
ax2.set_ylabel("Amplitude")

plt.tight_layout()
plt.show(block=False)

# === 특징 추출 함수 (리샘플링 포함) ===
def extract_features(y_audio):
    # 1. 리샘플링: 44100 → 22050
    y_audio = librosa.resample(y_audio, orig_sr=mic_sr, target_sr=model_sr)

    # 2. MFCC + ZCR 추출
    mfcc = librosa.feature.mfcc(y=y_audio, sr=model_sr, n_mfcc=n_mfcc, hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(y=y_audio, hop_length=hop_length)
    features = np.vstack([mfcc, zcr])  # (14, time)

    # 3. 입력 크기 정규화
    if features.shape[1] < max_len:
        features = np.pad(features, ((0, 0), (0, max_len - features.shape[1])), mode='constant')
    else:
        features = features[:, :max_len]

    # 4. reshape to (1, 86, 14, 1)
    return features.T[np.newaxis, ..., np.newaxis].astype(np.float32)

# === 마이크 콜백 ===
def audio_callback(indata, frames, time_info, status):
    global rolling_audio
    mono_input = indata[:, 0]
    rolling_audio = np.roll(rolling_audio, -len(mono_input))
    rolling_audio[-len(mono_input):] = mono_input

# === 예측 쓰레드 ===
def predict_thread():
    global rolling_audio, latest_pred, latest_label, latest_confidence
    segment_len = int(mic_sr * segment_duration)

    while True:
        time.sleep(1.0)  # 1초마다 추론
        audio_seg = rolling_audio[-segment_len:]
        x = extract_features(audio_seg)
        pred = model.predict(x, verbose=0)[0]  # (2,)
        p = float(pred[0])
        latest_pred = np.array([1 - p, p])
        label_idx = int(p > 0.5)
        latest_label = class_names[label_idx]
        latest_confidence = latest_pred[label_idx]
        print(f"\nPrediction: [{latest_label}] (Confidence: {latest_confidence:.2f})")

# === 마이크 스트리밍 시작 ===
print(" Real-time prediction started. Press Ctrl+C to stop.")
stream = sd.InputStream(
    device=device_index,
    callback=audio_callback,
    channels=1,
    samplerate=mic_sr,
    blocksize=int(mic_sr * 0.05)
)
stream.start()

threading.Thread(target=predict_thread, daemon=True).start()

# === 메인 루프 (시각화) ===
try:
    while True:
        line_wave.set_ydata(rolling_audio[-plot_len:])
        for i, bar in enumerate(bar_plot):
            bar.set_height(latest_pred[i])
        ax1.set_title(f"Prediction result: [{latest_label}] (Confidence: {latest_confidence:.2f})")
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.05)

except KeyboardInterrupt:
    print("\n Terminated by user.")
    stream.stop()
    plt.ioff()
    plt.close()
