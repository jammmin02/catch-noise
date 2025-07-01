import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import periodogram

# librosa 호환성 패치
np.complex = complex
np.float = float

# ===== 설정 =====
RAW_DIR = "../data/raw"
PLOT_DIR = "../data/plots"
OUTPUT_DIR = "data/processed"
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
segment_duration = 2  # 초 단위 세그먼트
sr_target = 22050     # 샘플링 레이트

limits = {
    "waveform": (-1.0, 1.0),
    "power": (0, 1e-4),
    "mfcc": (-500, 200),
    "rms": (0, 1),
    "zcr": (0, 0.5)
}

# ===== 시각화 함수 =====
def plot_segment_features(y_seg, sr, class_name):
    rms = librosa.feature.rms(y=y_seg)[0]
    rms /= np.max(rms) if np.max(rms) > 0 else 1
    zcr = librosa.feature.zero_crossing_rate(y_seg)[0]
    mfcc = librosa.feature.mfcc(y=y_seg, sr=sr, n_mfcc=13)
    S_power = librosa.feature.melspectrogram(y=y_seg, sr=sr)
    S_dB = librosa.power_to_db(S_power, ref=np.max)
    t_rms = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
    t_zcr = librosa.frames_to_time(np.arange(len(zcr)), sr=sr)
    f, Pxx = periodogram(y_seg, fs=sr)
    mask = f <= 11025

    fig, axs = plt.subplots(5, 1, figsize=(12, 20), constrained_layout=True)
    fig.suptitle(f"{class_name.capitalize()} - Sample Visualization", fontsize=18)

    axs[0].plot(np.linspace(0, len(y_seg)/sr, num=len(y_seg)), y_seg, color='orange')
    axs[0].set_title('Waveform')
    axs[0].set_ylim(limits["waveform"])

    axs[1].plot(f[mask], Pxx[mask], color='orange')
    axs[1].set_title('Power Spectrum (0–11kHz)')
    axs[1].set_ylim(limits["power"])
    axs[1].grid(True)

    img = librosa.display.specshow(mfcc, x_axis='time', sr=sr, ax=axs[2])
    axs[2].set_title('MFCC')
    axs[2].set_ylim(0, 13)
    img.set_clim(*limits["mfcc"])
    fig.colorbar(img, ax=axs[2])

    axs[3].plot(t_rms, rms, color='orange', label='RMS (Normalized)')
    axs[3].set_title('RMS Energy')
    axs[3].set_ylim(limits["rms"])
    axs[3].legend()

    axs[4].plot(t_zcr, zcr, color='green', label='ZCR')
    axs[4].set_title('Zero Crossing Rate')
    axs[4].set_ylim(limits["zcr"])
    axs[4].legend()

    save_path = os.path.join(PLOT_DIR, f"{class_name}.png")
    plt.savefig(save_path)
    plt.close()

# ===== 전처리 시작 =====
x_data, y_data = [], []
class_to_index = {}
visualized_classes = set()

for idx, class_name in enumerate(sorted(os.listdir(RAW_DIR))):
    class_path = os.path.join(RAW_DIR, class_name)
    if not os.path.isdir(class_path):
        continue
    class_to_index[class_name] = idx

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
            feature = np.concatenate([mfcc, zcr], axis=0)

            x_data.append(feature)
            y_data.append(idx)

            if class_name not in visualized_classes:
                plot_segment_features(y_seg, sr, class_name)
                visualized_classes.add(class_name)

# ===== 저장 =====
x_data = np.array(x_data)
y_data = np.array(y_data)
np.save(os.path.join(OUTPUT_DIR, "x.npy"), x_data)
np.save(os.path.join(OUTPUT_DIR, "y.npy"), y_data)

print(f" 전처리 완료: 총 {len(x_data)}개 세그먼트 저장됨")
print(f" 클래스 인덱스: {class_to_index}")
