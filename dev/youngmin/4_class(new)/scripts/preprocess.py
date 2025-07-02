import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import periodogram

# ===== librosa 호환성 패치 =====
np.complex = complex
np.float = float

# ===== 디렉토리 설정 (도커 기준 상대 경로) =====
RAW_DIR = "../data/raw"               # 원본 .wav 폴더 (클래스별 폴더 존재)
PLOT_DIR = "data/plots"            # 시각화 결과 저장
OUTPUT_DIR = "data/processed"      # .npy 전처리 결과 저장
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== 전처리 파라미터 설정 =====
segment_duration = 2              # 각 세그먼트 길이 (초)
sr_target = 22050                 # 고정 샘플링 레이트

# ===== 시각화 고정 범위 설정 =====
limits = {
    "waveform": (-1.0, 1.0),
    "power": (0, 1e-4),
    "mfcc": (-500, 200),
    "rms": (0, 1),
    "zcr": (0, 0.5)
}

# ===== 전처리 시작 =====
x_data, y_data = [], []
class_to_index = {}
visualized_classes = set()

# ===== 클래스 필터링 및 수동 인덱스 부여 =====
target_classes = ["person", "coughs", "natural"]
class_to_index = {name: i for i, name in enumerate(target_classes)}

for class_name in sorted(os.listdir(RAW_DIR)):
    if class_name not in target_classes:
        print(f"[SKIP] '{class_name}' 클래스 제외됨")
        continue

    class_path = os.path.join(RAW_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    idx = class_to_index[class_name]

    for file in sorted(os.listdir(class_path)):
        if not file.endswith(".wav"):
            continue
        file_path = os.path.join(class_path, file)
        y, sr = librosa.load(file_path, sr=sr_target)
        segment_samples = int(segment_duration * sr)

        for i in range(0, len(y), segment_samples):
            y_seg = y[i:i + segment_samples]
            if len(y_seg) < segment_samples:
                continue

            mfcc = librosa.feature.mfcc(y=y_seg, sr=sr, n_mfcc=13)
            zcr = librosa.feature.zero_crossing_rate(y_seg)[0].reshape(1, -1)
            try:
                feature = np.concatenate([mfcc, zcr], axis=0)
            except:
                continue

            x_data.append(feature)
            y_data.append(idx)

            if class_name not in visualized_classes:
                visualized_classes.add(class_name)


# ===== 결과 저장 =====
x_data = np.array(x_data)
y_data = np.array(y_data)
np.save(os.path.join(OUTPUT_DIR, "x.npy"), x_data)
np.save(os.path.join(OUTPUT_DIR, "y.npy"), y_data)

print(f"전처리 완료: 총 {len(x_data)}개 세그먼트 저장됨")
print(f"클래스 인덱스 매핑: {class_to_index}")
