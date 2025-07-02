# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import torch.nn as nn
import librosa
import sounddevice as sd
import matplotlib.pyplot as plt
import matplotlib
from time import sleep

# TkAgg 백엔드 설정 (Tkinter 기반 GUI 사용)
matplotlib.use('TkAgg')

# ===== 설정 =====
SAMPLE_RATE = 22050
DURATION = 2.0
N_MFCC = 13
INPUT_SHAPE = (1, 14, 87)
MODEL_PATH = "scripts/model/cnn_audio_classifier.pth"
CLASS_NAMES = ["person", "cough", "natural"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 마이크 자동 설정 =====
try:
    devices = sd.query_devices()
    input_devices = [i for i, d in enumerate(devices) if d['max_input_channels'] > 0]
    if not input_devices:
        raise RuntimeError("입력 가능한 마이크 디바이스가 없습니다.")
    sd.default.device = (input_devices[0], None)
    print(f"[INFO] 마이크 디바이스: {devices[input_devices[0]]['name']}")
except Exception as e:
    print(f"[ERROR] 마이크 설정 실패: {e}")
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
try:
    print("[INFO] 모델 로딩 중...")
    model = SimpleCNN(input_shape=INPUT_SHAPE, num_classes=len(CLASS_NAMES)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("[INFO] 모델 로딩 완료")
except Exception as e:
    print(f"[ERROR] 모델 로딩 실패: {e}")
    exit(1)

# ===== 예측 함수 =====
def predict(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    zcr = librosa.feature.zero_crossing_rate(audio)[0].reshape(1, -1)
    feature = np.concatenate([mfcc, zcr], axis=0)
    if feature.shape[1] < 87:
        feature = np.pad(feature, ((0, 0), (0, 87 - feature.shape[1])), mode='constant')
    else:
        feature = feature[:, :87]
    x = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(x)
        probs = torch.softmax(output, dim=1).cpu().numpy().flatten()
        pred = np.argmax(probs)
    return CLASS_NAMES[pred], probs

# ===== 게이지 시각화 함수 =====
def update_gauge_bars(probabilities, class_names):
    plt.clf()
    bars = plt.barh(class_names, probabilities, color='mediumseagreen')  # 초록색 가로 막대
    plt.xlim(0, 1)
    plt.xlabel("Confidence")
    plt.title("실시간 예측 결과 (Confidence)")

    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{probabilities[i]:.2f}", va='center')
    
    plt.tight_layout()
    plt.draw()
    plt.pause(0.01)


# ===== 시각화 초기화 =====
plt.ion()
plt.figure(figsize=(6, 3))
plt.show(block=False)

# ===== 실시간 루프 =====
try:
    while True:
        print("⏺ 2초간 녹음 중...")
        audio = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()
        audio = audio.flatten()
        try:
            pred_class, pred_probs = predict(audio)
            print(f"[예측 결과]: {pred_class}")
            update_gauge_bars(pred_probs, CLASS_NAMES)
        except Exception as e:
            print(f"[ERROR] 예측 실패: {e}")
        sleep(1)
except KeyboardInterrupt:
    print("\n[INFO] 실시간 예측 종료됨.")
    plt.ioff()
    plt.close()
