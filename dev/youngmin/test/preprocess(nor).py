import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

# ===== 설정 =====
input_root = 'audio'
output_root = 'outputs_normalized/visuals'
label_names = ['noisy', 'non_noisy']
sampling_rate = 22050
n_mfcc = 13
hop_length = 512
duration = 2.0
fixed_frame_len = int((sampling_rate / hop_length) * duration)
vmin, vmax = 0, 1  # 정규화는 0~1 고정

# ===== 전처리 + 정규화 함수 =====
def extract_mfcc_zcr_normalized(y):
    mfcc = librosa.feature.mfcc(y=y, sr=sampling_rate, n_mfcc=n_mfcc, hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)
    features = np.vstack([mfcc, zcr])  # (14, T)

    # 시간 길이 고정
    if features.shape[1] < fixed_frame_len:
        features = np.pad(features, ((0, 0), (0, fixed_frame_len - features.shape[1])), mode='constant')
    else:
        features = features[:, :fixed_frame_len]

    # 정규화 (0~1)
    scaler = MinMaxScaler()
    features_norm = scaler.fit_transform(features.T).T  # (14, T)

    return features_norm, zcr[0]

# ===== 시각화 함수 =====
def draw_3part_figure(y, features_scaled, zcr_curve, title, save_path):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)

    # 1. Waveform
    axes[0].plot(y, color='gray')
    axes[0].set_title("Waveform")
    axes[0].set_ylabel("Amplitude")

    # 2. MFCC + ZCR 히트맵
    im = axes[1].imshow(features_scaled, aspect='auto', origin='lower', cmap='coolwarm', vmin=vmin, vmax=vmax)
    axes[1].set_title("MFCC + ZCR (Normalized)")
    axes[1].set_ylabel("Feature Index")
    axes[1].set_yticks(np.arange(14))
    axes[1].set_yticklabels([f"MFCC{i}" for i in range(n_mfcc)] + ["ZCR"])
    fig.colorbar(im, ax=axes[1])

    # 3. ZCR 곡선
    axes[2].plot(zcr_curve, color='purple')
    axes[2].set_ylim(0, 0.5)
    axes[2].set_title("ZCR Curve")
    axes[2].set_xlabel("Frame Index")
    axes[2].set_ylabel("ZCR")

    # 저장
    fig.suptitle(title, fontsize=14)
    plt.savefig(save_path)
    plt.close()
    print(f"✅ 저장됨: {save_path}")

# ===== 전체 실행 =====
if __name__ == "__main__":
    for label in label_names:
        input_dir = os.path.join(input_root, label)
        output_dir = os.path.join(output_root, label)
        os.makedirs(output_dir, exist_ok=True)

        files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.wav', '.mp3', '.m4a'))]

        print(f"🔍 {label.upper()} - {len(files)}개 파일 처리 중...")
        for fname in tqdm(files, desc=f"[{label}]"):
            filepath = os.path.join(input_dir, fname)
            try:
                y, _ = librosa.load(filepath, sr=sampling_rate, duration=duration)
                features_norm, zcr_curve = extract_mfcc_zcr_normalized(y)
                base_name = os.path.splitext(fname)[0]
                save_path = os.path.join(output_dir, f"{base_name}.png")
                draw_3part_figure(y, features_norm, zcr_curve, f"{base_name} - {label}", save_path)
            except Exception as e:
                print(f"❗ 오류 발생: {fname} → {e}")

    print("\n✅ 모든 정규화 시각화 완료!")
