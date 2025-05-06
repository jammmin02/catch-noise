import sounddevice as sd
import librosa
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import threading
import time

# 🎛 설정
sr = 22050
segment_duration = 2.0  # 예측용 segment 길이 (초)
n_mfcc = 13
hop_length = 512
frame_per_second = sr / hop_length
max_len = int(frame_per_second * segment_duration)
class_names = ['non_noisy', 'noisy']
model_path = "C:/Users/USER/.aCode/catch-noise/dev/hyochan/2class_predict_model/dataset/outputs/cnn_lstm/cnn_lstm_model.keras"
model = load_model(model_path)

# 🔊 오디오 버퍼
rolling_audio = np.zeros(int(segment_duration * sr), dtype=np.float32)

# 🧠 예측 결과 저장용 변수 (메인 루프에서 접근)
latest_pred = [0.5, 0.5]
latest_label = "non_noisy"
latest_confidence = 0.5

# 🎨 시각화 초기화
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

# (1) 예측 결과 막대 그래프 (상단)
bar_plot = ax1.bar(class_names, latest_pred, color=["skyblue", "salmon"])
ax1.set_ylim(0, 1)
ax1.set_title("Prediction results")
ax1.set_ylabel("Probability")

# (2) 실시간 오디오 파형 (하단)
plot_len = int(sr * 0.5)
line_wave, = ax2.plot(np.zeros(plot_len))
ax2.set_ylim(-1, 1)
ax2.set_title("Enter real-time microphone")
ax2.set_xlabel("Sample")
ax2.set_ylabel("Amplitude")

# 🎛 특징 추출 함수
def extract_features(y_audio):
    mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(y=y_audio, hop_length=hop_length)
    features = np.vstack([mfcc, zcr])
    if features.shape[1] < max_len:
        features = np.pad(features, ((0, 0), (0, max_len - features.shape[1])), mode='constant')
    else:
        features = features[:, :max_len]
    return features.T[np.newaxis, ..., np.newaxis]  # (1, max_len, 14, 1)

# 🎤 마이크 콜백 (0.05초 단위)
def audio_callback(indata, frames, time_info, status):
    global rolling_audio
    indata = indata[:, 0]
    rolling_audio = np.roll(rolling_audio, -len(indata))
    rolling_audio[-len(indata):] = indata

# 🧠 예측 쓰레드 (1초마다 중첩된 2초 구간 예측)
def predict_thread():
    global rolling_audio, latest_pred, latest_label, latest_confidence
    segment_len = int(sr * segment_duration)  # 2초 segment

    while True:
        time.sleep(1.0)  # 1초마다 예측 (50% 중첩)
        if len(rolling_audio) >= segment_len:
            audio_seg = rolling_audio[-segment_len:]
            x = extract_features(audio_seg)
            pred = model.predict(x, verbose=0)[0]
            label_idx = np.argmax(pred)
            latest_pred = pred
            latest_label = class_names[label_idx]
            latest_confidence = pred[label_idx]
            print(f"\n🔁 중첩 예측: [{latest_label}] (신뢰도: {latest_confidence:.2f})")

# 🎬 실행
print("🎙️ 실시간 파형 + 중첩 예측 시작 (Ctrl+C로 종료)...")
stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=sr, blocksize=int(sr * 0.05))
stream.start()

# 🧵 예측 쓰레드 시작
threading.Thread(target=predict_thread, daemon=True).start()

# 📈 메인 루프: 실시간 파형 + 예측 그래프 시각화
try:
    while True:
        # 실시간 파형 업데이트
        line_wave.set_ydata(rolling_audio[-plot_len:])

        # 예측 그래프 업데이트
        for i, bar in enumerate(bar_plot):
            bar.set_height(latest_pred[i])
        ax1.set_title(f"🔮 예측 결과: [{latest_label}] (신뢰도: {latest_confidence:.2f})")

        # 전체 플롯 갱신
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.05)

except KeyboardInterrupt:
    print("\n🛑 종료됨.")
    stream.stop()
    plt.ioff()
    plt.close()
