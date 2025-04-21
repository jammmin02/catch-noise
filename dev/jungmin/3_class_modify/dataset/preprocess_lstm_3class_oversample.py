import os
import librosa
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import random
import uuid

# 설정
base_dir = 'data'
output_dir = 'dev/jungmin/3_class_modify/outputs/cnn_lstm'
sr = 22050
n_mfcc = 13
hop_length = 512
segment_duration = 3.0
save_visuals = True

frame_per_second = sr / hop_length
max_len = int(frame_per_second * segment_duration)

label_names = ['neutral', 'non_noisy', 'noisy']
label_map = {name: idx for idx, name in enumerate(label_names)}

X, y, logs = [], [], []

# 폴더 생성
for label_name in label_names:
    os.makedirs(os.path.join(output_dir, 'visuals', label_name), exist_ok=True)

def convert_to_wav(src_path, dst_path):
    if not os.path.exists(dst_path):
        command = ['ffmpeg', '-y', '-i', src_path, '-ac', '1', '-ar', str(sr), dst_path]
        subprocess.run(command, check=True)

# 데이터 처리
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
            total_duration = librosa.get_duration(path=file_path)
        except:
            print(f"❗ duration 읽기 실패: {file_path}")
            continue

        segment_count = int(total_duration // segment_duration)
        if segment_count == 0:
            print(f"⏩ 짧은 오디오 스킵: {file_path}")
            continue

        for i in range(segment_count):
            offset = i * segment_duration
            try:
                y_audio, _ = librosa.load(file_path, sr=sr, offset=offset, duration=segment_duration)
            except:
                print(f"❗ load 실패: {file_path} (segment {i})")
                continue

            mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
            zcr = librosa.feature.zero_crossing_rate(y=y_audio, hop_length=hop_length)
            zcr_mean = np.mean(zcr)  # ZCR 평균값만 사용 (1값)

            zcr_feature = np.full((1, mfcc.shape[1]), zcr_mean)
            features = np.vstack([mfcc, zcr_feature])

            if features.shape[1] < max_len:
                features = np.pad(features, ((0, 0), (0, max_len - features.shape[1])), mode='constant')
            else:
                features = features[:, :max_len]

            X.append(features.T.astype(np.float32))
            y.append(label)

            log_entry = {
                "file": file_name,
                "segment": i + 1,
                "label": label_name,
                "uuid": str(uuid.uuid4())
            }
            logs.append(log_entry)

            if save_visuals:
                save_filename = f"{label_name}_{base_filename}_seg{i+1}_{log_entry['uuid']}.png"
                save_path = os.path.join(output_dir, 'visuals', label_name, save_filename)

                fig, ax = plt.subplots(3, 1, figsize=(12, 10))

                # 1️⃣ Waveform
                librosa.display.waveshow(y_audio, sr=sr, ax=ax[0])
                ax[0].set_title(f"Waveform - Segment {i+1}")
                ax[0].set_xlabel("Time (s)")
                ax[0].set_ylabel("Amplitude")

                # 2️⃣ MFCC
                img = librosa.display.specshow(mfcc, x_axis="time", sr=sr, hop_length=hop_length, ax=ax[1])
                ax[1].set_title("MFCC")
                fig.colorbar(img, ax=ax[1], format="%+2.f dB")

                # 3️⃣ ZCR
                ax[2].plot(np.linspace(0, segment_duration, zcr.shape[1]), zcr[0])
                ax[2].set_title("Zero Crossing Rate")
                ax[2].set_xlabel("Time (s)")
                ax[2].set_ylabel("ZCR")

                plt.tight_layout()
                plt.savefig(save_path)
                plt.close()
                
# 🎯 클래스 불균형 처리: oversampling
print("📊 클래스 균형 맞추는 중...")
data_by_label = {i: [] for i in range(len(label_names))}
for feat, lab in zip(X, y):
    data_by_label[lab].append(feat)

max_len_class = max(len(v) for v in data_by_label.values())

X_balanced, y_balanced = [], []
for label, feats in data_by_label.items():
    if len(feats) == 0:
        print(f"⚠️ 클래스 '{label_names[label]}'에 유효한 샘플이 없습니다. 건너뜁니다.")
        continue
    repeats = max_len_class // len(feats)
    remainder = max_len_class % len(feats)

    # 데이터 복사 및 일부 랜덤 추가
    balanced_feats = feats * repeats + random.sample(feats, remainder)
    X_balanced.extend(balanced_feats)
    y_balanced.extend([label] * max_len_class)

# 📊 시각화: 오버샘플링 전후 클래스별 데이터 수 비교
original_counts = {label_names[k]: len(v) for k, v in data_by_label.items()}
oversampled_counts = {label_names[k]: max_len_class for k in data_by_label.keys() if len(data_by_label[k]) > 0}

labels = list(original_counts.keys())
x = range(len(labels))
width = 0.35

fig, ax = plt.subplots()
ax.bar([i - width/2 for i in x], original_counts.values(), width, label='Original', color='skyblue')
ax.bar([i + width/2 for i in x], oversampled_counts.values(), width, label='Oversampled', color='salmon')

ax.set_ylabel('Sample Count')
ax.set_title('Sample Count by Class (Before vs After Oversampling)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.tight_layout()

# 이미지 파일로 저장
plt.savefig(os.path.join(output_dir, "oversampling_visualization.png"))
plt.show()

# 저장
os.makedirs(output_dir, exist_ok=True)
np.save(os.path.join(output_dir, "X_lstm.npy"), np.array(X_balanced, dtype=np.float32))
np.save(os.path.join(output_dir, "y_lstm.npy"), np.array(y_balanced, dtype=np.int32))
pd.DataFrame(logs).to_csv(os.path.join(output_dir, "segment_logs.csv"), index=False)

# 결과 출력
print("✅ segment_duration:", segment_duration)
print("✅ max_len:", max_len)
print("✅ X shape:", np.array(X_balanced).shape)
print("✅ y shape:", np.array(y_balanced).shape)
print("📁 저장 완료:", output_dir)
print("📁 시각화 저장 완료:", os.path.join(output_dir, "oversampling_visualization.png"))