import os
import librosa
import numpy as np
import subprocess
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
import random

# 설정
base_dir = 'hyochan/pc/data'
output_dir = 'hyochan/pc/dataset/outputs/cnn_lstm'
sr = 22050
n_mfcc = 13
hop_length = 512
segment_duration = 2.0
save_visuals = True

frame_per_second = sr / hop_length
max_len = int(frame_per_second * segment_duration)

X, y = [], []

label_names = ['non_noisy', 'noisy']
label_map = {name: idx for idx, name in enumerate(label_names)}

# 🔁 ffmpeg 변환 함수
def convert_to_wav(src_path, dst_path):
    if not os.path.exists(dst_path):
        command = ['ffmpeg', '-y', '-i', src_path, '-ac', '1', '-ar', str(sr), dst_path]
        subprocess.run(command, check=True)

# 📊 클래스 분포 시각화 저장 함수
def save_class_distribution_graph(original_y, oversampled_y, label_names, save_path):
    before_counter = Counter(original_y)
    after_counter = Counter(oversampled_y)
    before_counts = [before_counter.get(i, 0) for i in range(len(label_names))]
    after_counts = [after_counter.get(i, 0) for i in range(len(label_names))]

    x = np.arange(len(label_names))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, before_counts, width, label='Before', color='gray')
    plt.bar(x + width/2, after_counts, width, label='After', color='skyblue')
    plt.xticks(x, label_names)
    plt.ylabel('Sample Count')
    plt.title('Class Distribution Before vs After Oversampling')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📊 클래스 분포 그래프 저장 완료: {save_path}")

# 🧠 오버샘플링 함수
def oversample_data(X, y):
    print("\n📊 오버샘플링 시작...")
    counter = Counter(y)
    if not counter:
        raise ValueError("y가 비어있습니다. 오버샘플링 불가능합니다.")
    max_count = max(counter.values())
    new_X, new_y = list(X), list(y)

    for label, count in counter.items():
        gap = max_count - count
        if gap > 0:
            print(f"🔁 {label_names[label]} 클래스 오버샘플링: +{gap}개")
            indices = [i for i, lbl in enumerate(y) if lbl == label]
            for _ in range(gap):
                idx = random.choice(indices)
                new_X.append(X[idx])
                new_y.append(y[idx])
    return new_X, new_y

# 📁 시각화 폴더 생성
for label_name in label_names:
    os.makedirs(os.path.join(output_dir, 'visuals', label_name), exist_ok=True)

# 🎧 전처리 루프 시작
for label_name in label_names:
    folder_path = os.path.join(base_dir, label_name)
    label = label_map[label_name]

    if not os.path.exists(folder_path):
        print(f"⚠️ 폴더 없음: {folder_path} → 건너뜀")
        continue

    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.wav', '.mp4', '.m4a', '.mp3'))]
    print(f"🎧 [{label_name}] 총 {len(files)}개 파일 처리 중...")

    for file_name in tqdm(files, desc=f"[{label_name}] 전처리", unit="file"):
        file_path = os.path.join(folder_path, file_name)

        if file_name.lower().endswith(('.mp4', '.m4a', '.mp3')):
            wav_name = os.path.splitext(file_name)[0] + '.wav'
            wav_path = os.path.join(folder_path, wav_name)
            convert_to_wav(file_path, wav_path)
            file_path = wav_path

        base_filename = os.path.splitext(file_name)[0]
        try:
            y_audio_all, _ = librosa.load(file_path, sr=sr)
            total_duration = librosa.get_duration(y=y_audio_all, sr=sr)
        except Exception as e:
            print(f"❗ duration 읽기 실패: {file_path} - {e}")
            continue

        segment_count = int(total_duration // segment_duration)
        if segment_count == 0:
            print(f"⚠️ usable segment 없음 (duration too short): {file_path}")
            continue

        for i in range(segment_count):
            offset = i * segment_duration
            try:
                y_audio, _ = librosa.load(file_path, sr=sr, offset=offset, duration=segment_duration)
            except Exception as e:
                print(f"❗ load 실패: {file_path} (segment {i}) - {e}")
                continue

            mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
            zcr = librosa.feature.zero_crossing_rate(y=y_audio, hop_length=hop_length)
            features = np.vstack([mfcc, zcr])

            if np.isnan(features).any():
                print(f"⚠️ NaN 존재: {file_path} (segment {i})")
                continue

            if features.shape[1] < max_len:
                features = np.pad(features, ((0, 0), (0, max_len - features.shape[1])), mode='constant')
            else:
                features = features[:, :max_len]

            X.append(features.T)
            y.append(label)

            if save_visuals:
                save_path = os.path.join(output_dir, 'visuals', label_name, f"{base_filename}_seg{i+1}.png")
                plt.figure(figsize=(10, 4))
                plt.imshow(features, aspect='auto', origin='lower', cmap='coolwarm')
                plt.title(f"{base_filename}_seg{i+1} - {label_name}")
                plt.xlabel("Frame")
                plt.ylabel("Feature Index (MFCC+ZCR)")
                plt.colorbar()
                plt.tight_layout()
                plt.savefig(save_path)
                plt.close()

# 🔁 오버샘플링 + 클래스 분포 시각화 저장
original_y = y.copy()
try:
    X, y = oversample_data(X, y)
    save_class_distribution_graph(original_y, y, label_names, os.path.join(output_dir, 'class_distribution.png'))
except ValueError as e:
    print(f"❌ 오버샘플링 실패: {e}")
    exit(1)

# 💾 저장
os.makedirs(output_dir, exist_ok=True)
np.save(os.path.join(output_dir, "X_lstm.npy"), np.array(X))
np.save(os.path.join(output_dir, "y_lstm.npy"), np.array(y))

print("✅ segment_duration:", segment_duration)
print("✅ Calculated max_len:", max_len)
print("✅ X shape (LSTM용):", np.array(X).shape)
print("✅ y shape:", np.array(y).shape)
print("📁 저장 완료:", output_dir)
