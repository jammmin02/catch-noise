import os
import librosa
import numpy as np
import subprocess
from tqdm import tqdm

# ✅ 경로 설정
base_dir = 'dev/youngmin/test_data'
output_dir = 'dev/youngmin/test_outputs'
sr = 22050
n_mfcc = 13
hop_length = 512
segment_duration = 1.0

frame_per_second = sr / hop_length
max_len = int(frame_per_second * segment_duration)

X, y = [], []
label_names = ['quiet', 'neutral', 'noisy']
label_map = {name: idx for idx, name in enumerate(label_names)}

def convert_to_wav(src_path, dst_path):
    if not os.path.exists(dst_path):
        command = ['ffmpeg', '-y', '-i', src_path, '-ac', '1', '-ar', str(sr), dst_path]
        subprocess.run(command, check=True)

os.makedirs(output_dir, exist_ok=True)

for label_name in label_names:
    folder_path = os.path.join(base_dir, label_name)
    label = label_map[label_name]

    if not os.path.exists(folder_path):
        continue

    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.wav', '.mp3', '.m4a', '.mp4'))]

    for file_name in tqdm(files, desc=f"[{label_name}]"):
        file_path = os.path.join(folder_path, file_name)

        if file_name.lower().endswith(('.mp3', '.mp4', '.m4a')):
            wav_name = os.path.splitext(file_name)[0] + '.wav'
            wav_path = os.path.join(folder_path, wav_name)
            convert_to_wav(file_path, wav_path)
            file_path = wav_path

        try:
            total_duration = librosa.get_duration(path=file_path)
        except:
            continue

        segment_count = int(total_duration // segment_duration)

        for i in range(segment_count):
            offset = i * segment_duration
            try:
                y_audio, _ = librosa.load(file_path, sr=sr, offset=offset, duration=segment_duration)
            except:
                continue

            mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
            zcr = librosa.feature.zero_crossing_rate(y=y_audio, hop_length=hop_length)
            features = np.vstack([mfcc, zcr])

            if features.shape[1] < max_len:
                features = np.pad(features, ((0, 0), (0, max_len - features.shape[1])), mode='constant')
            else:
                features = features[:, :max_len]

            X.append(features.T)
            y.append(label)

# ✅ 저장
np.save(os.path.join(output_dir, "X_test.npy"), np.array(X))
np.save(os.path.join(output_dir, "y_test.npy"), np.array(y))
print("✅ 테스트셋 전처리 완료!")
