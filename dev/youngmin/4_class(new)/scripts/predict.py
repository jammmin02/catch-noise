import os
import numpy as np
import torch
import torch.nn as nn
import librosa
import sounddevice as sd
from time import sleep

# ===== 설정 =====
SAMPLE_RATE = 22050
DURATION = 1  # 1초 단위 예측
N_MFCC = 13
MODEL_PATH = "model/cnn_audio_classifier.pth"
CLASS_NAMES = ["person", "cough", "laugh", "natural"]  # 클래스 이름 순서 중요
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 입력 디바이스 자동 선택 =====
try:
    devices = sd.query_devices()
    input_devices = [i for i, d in enumerate(devices) if d['max_input_channels'] > 0]
    if not input_devices:
        raise RuntimeError("입력 가능한 마이크 디바이스가 없습니다.")
    sd.default.device = input_devices[0]
    print(f"[INFO] 입력 디바이스 설정됨: {devices[sd.default.device]['name']}")
except Exception as e:
    print(f"[ERROR] 마이크 디바이스 설정 실패: {e}")
    exit(1)

# ===== 모델 정의 =====
class SimpleCNN(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            out = self.conv(dummy)
            self.flattened_dim = out.view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

# ===== 모델 로딩 =====
print("[INFO] 모델 로딩 중...")
model = SimpleCNN(input_shape=(1, 14, 87), num_classes=len(CLASS_NAMES)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("[INFO] 모델 로딩 완료")

# ===== 예측 함수 =====
def predict(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    zcr = librosa.feature.zero_crossing_rate(audio)[0].reshape(1, -1)
    feature = np.concatenate([mfcc, zcr], axis=0)
    x = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1, 1, 14, T)
    with torch.no_grad():
        output = model(x)
        pred = output.argmax(1).item()
    return CLASS_NAMES[pred]

# ===== 실시간 예측 루프 =====
print("🎙 실시간 마이크 입력 시작 (Ctrl+C로 중단)")

try:
    while True:
        print("⏺ 녹음 중...")
        audio = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()
        audio = audio.flatten()

        pred_class = predict(audio)
        print(f" 예측 결과: {pred_class}")
        sleep(0.5)

except KeyboardInterrupt:
    print("\n[INFO] 예측 종료됨.")
