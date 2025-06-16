import os
import sounddevice as sd
import librosa
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # ✅ 안정적인 GUI 백엔드 사용 (Windows 추천)
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import threading
import time

# === 설정 ===
mic_sr = 44100
model_sr = 22050
segment_duration = 2.0
n_mfcc = 13
hop_length = 512
max_len = 86
class_names = ['non_noisy', 'noisy']

# ✅ PyTorch 모델 경로
model_path = f"C:/Users/USER/.aCode/catch-noise/dev/hyochan/pytorch/pc/dataset/outputs/cnn_lstm/cnn_lstm_model.pth"

# ✅ CNN+LSTM 모델 정의 (학습 때와 동일하게 유지)
class CNN_LSTM(nn.Module):
    def __init__(self, input_height, input_width):
        super(CNN_LSTM, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)

        self.height = input_height // 4
        self.width = input_width // 4
        self.lstm_input_size = self.width * 64

        self.reshape_for_lstm = lambda x: x.permute(0, 2, 1, 3).contiguous().view(-1, self.height, self.lstm_input_size)

        self.lstm = nn.LSTM(self.lstm_input_size, 64, batch_first=True)
        self.fc1 = nn.Linear(64, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.reshape_for_lstm(x)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x

# ✅ 모델 초기화 및 로드
model = CNN_LSTM(input_height=max_len, input_width=n_mfcc + 1)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

# === 마이크 장치 자동 탐색 ===
def find_input_device(name_keyword=None):
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        if (name_keyword is None or name_keyword.lower() in dev["name"].lower()) and dev["max_input_channels"] > 0:
            print(f"Selected microphone: {dev['name']} (index {idx})")
            return idx, dev["max_input_channels"]
    raise RuntimeError("Microphone not found.")

# ✅ 여기서 디바이스 수동으로 잡아라 (자동탐색 실패 대비)
device_index, device_channels = find_input_device()

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
    y_audio = librosa.resample(y_audio, orig_sr=mic_sr, target_sr=model_sr)
    mfcc = librosa.feature.mfcc(y=y_audio, sr=model_sr, n_mfcc=n_mfcc, hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(y=y_audio, hop_length=hop_length)
    features = np.vstack([mfcc, zcr])

    if features.shape[1] < max_len:
        features = np.pad(features, ((0, 0), (0, max_len - features.shape[1])), mode='constant')
    else:
        features = features[:, :max_len]

    features = features.T[np.newaxis, ..., np.newaxis].astype(np.float32)
    tensor = torch.tensor(features).permute(0, 3, 1, 2)
    return tensor

# === 마이크 콜백 ===
def audio_callback(indata, frames, time_info, status):
    global rolling_audio
    mono_input = indata[:, 0]
    rolling_audio = np.roll(rolling_audio, -len(mono_input))
    rolling_audio[-len(mono_input):] = mono_input

# === 예측 쓰레드 (데몬 제거 → 안정성 확보)
def predict_thread():
    global rolling_audio, latest_pred, latest_label, latest_confidence
    segment_len = int(mic_sr * segment_duration)

    while True:
        time.sleep(1.0)
        audio_seg = rolling_audio[-segment_len:]
        x = extract_features(audio_seg)

        with torch.no_grad():
            output = model(x).squeeze().item()
        p = float(output)
        latest_pred = np.array([1 - p, p])
        label_idx = int(p > 0.5)
        latest_label = class_names[label_idx]
        latest_confidence = latest_pred[label_idx]
        print(f"Prediction: [{latest_label}] (Confidence: {latest_confidence:.2f})")

# === 마이크 스트리밍 시작 ===
print("Real-time prediction started. Press Ctrl+C to stop.")

stream = sd.InputStream(
    device=device_index,
    callback=audio_callback,
    channels=1,
    samplerate=mic_sr,
    blocksize=int(mic_sr * 0.05)
)
stream.start()

# ✅ 데몬 쓰레드 제거 → 안정성 상승
thread = threading.Thread(target=predict_thread)
thread.start()

# === 메인 루프 (시각화 유지)
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
    print("\nTerminated by user.")
    stream.stop()
    plt.ioff()
    plt.close()
