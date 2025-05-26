import os
import librosa
import numpy as np
import subprocess
import matplotlib.pyplot as plt
from tqdm import tqdm

# ✅ 설정
input_root = 'dev/youngmin/raw_data'  # 오디오 원본
output_root = 'dev/youngmin/outputs'  # 결과 저장 폴더
sampling_rate = 22050
n_mfcc = 13
hop_length = 512
segment_duration_sec = 1.0  # ⚠️ 1초 단위
save_visuals = True

frames_per_second = sampling_rate / hop_length
target_frame_len = int(frames_per_second * segment_duration_sec)

X_features, y_labels = [], []

label_names = ['quiet', 'neutral', 'noisy']
label_to_index = {name: idx for idx, name in enumerate(label_names)}

def convert_to_wav(input_path, output_path):
    if not os.path.exists(output_path):
        command = ['ffmpeg', '-y', '-i', input_path, '-ac', '1', '-ar', str(sampling_rate), output_path]
        subprocess.run(command, check=True)

# 📁 출력 디렉토리 준비
for label in label_names:
    vis_path = os.path.join(output_root, 'visuals', label)
    os.makedirs(vis_path, exist_ok=True)

# 📦 전처리 시작
for label in label_names:
    folder_path = os.path.join(input_root, label)
    label_index = label_to_index[label]

    if not os.path.exists(folder_path):
        print(f"⚠️ 경로 없음: {folder_path}")
        continue

    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.wav', '.mp3', '.m4a', '.mp4'))]
    print(f"🎧 [{label}] 총 {len(files)}개 파일 처리 중...")

    for fname in tqdm(files, desc=f"[{label}] 전처리", unit="file"):
        filepath = os.path.join(folder_path, fname)

        if fname.lower().endswith(('.mp3', '.mp4', '.m4a')):
            wav_name = os.path.splitext(fname)[0] + '.wav'
            wav_path = os.path.join(folder_path, wav_name)
            convert_to_wav(filepath, wav_path)
            filepath = wav_path

        base_name = os.path.splitext(fname)[0]
        try:
            duration = librosa.get_duration(path=filepath)
        except:
            print(f"❗ duration 읽기 실패: {filepath}")
            continue

        segment_count = int(duration // segment_duration_sec)

        for i in range(segment_count):
            offset_sec = i * segment_duration_sec
            try:
                audio_data, _ = librosa.load(filepath, sr=sampling_rate, offset=offset_sec, duration=segment_duration_sec)
            except:
                print(f"❗ 로드 실패: {filepath} (세그먼트 {i})")
                continue

            mfcc = librosa.feature.mfcc(y=audio_data, sr=sampling_rate, n_mfcc=n_mfcc, hop_length=hop_length)
            zcr = librosa.feature.zero_crossing_rate(y=audio_data, hop_length=hop_length)
            features = np.vstack([mfcc, zcr])

            if features.shape[1] < target_frame_len:
                features = np.pad(features, ((0, 0), (0, target_frame_len - features.shape[1])), mode='constant')
            else:
                features = features[:, :target_frame_len]

            X_features.append(features.T)
            y_labels.append(label_index)

            if save_visuals:
                vis_path = os.path.join(output_root, 'visuals', label, f"{base_name}_seg{i+1}.png")
                plt.figure(figsize=(10, 4))
                plt.imshow(features, aspect='auto', origin='lower', cmap='coolwarm')
                plt.title(f"{base_name}_seg{i+1} - {label}")
                plt.xlabel("Frame")
                plt.ylabel("Feature Index (MFCC+ZCR)")
                plt.colorbar()
                plt.tight_layout()
                plt.savefig(vis_path)
                plt.close()

# ✅ 저장
os.makedirs(output_root, exist_ok=True)
np.save(os.path.join(output_root, "X_lstm.npy"), np.array(X_features))
np.save(os.path.join(output_root, "y_lstm.npy"), np.array(y_labels))

# ✅ 출력
print("✅ segment_duration (초):", segment_duration_sec)
print("✅ target_frame_len:", target_frame_len)
print("✅ X shape:", np.array(X_features).shape)
print("✅ y shape:", np.array(y_labels).shape)
print("📁 저장 완료:", output_root)
