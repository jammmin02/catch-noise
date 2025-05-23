# 📦 라이브러리 불러오기
import os
import librosa
import numpy as np
import subprocess
import matplotlib.pyplot as plt
from tqdm import tqdm

# 🔧 설정
base_dir = 'data'  # 오디오 데이터 폴더 경로 (data/quiet, data/loud)
output_dir = 'outputs'  # 결과 저장 폴더
sr = 22050  # 샘플링 레이트
n_mfcc = 13  # MFCC 계수 수
hop_length = 512  # 프레임 간 간격
segment_duration = 5.0  # 세그먼트 길이 (초)
save_visuals = True  # 스펙트로그램 시각화 여부

# 🎯 프레임 수 계산
frame_per_second = sr / hop_length
max_len = int(frame_per_second * segment_duration)

# 🎯 결과 저장용 리스트 초기화
X, y = [], []

# 🏷️ 라벨 설정
label_names = ['quiet', 'loud']
label_map = {name: idx for idx, name in enumerate(label_names)}

# 🎧 ffmpeg를 사용해 mp3/mp4/m4a를 wav로 변환하는 함수
def convert_to_wav(src_path, dst_path):
    if not os.path.exists(dst_path):
        command = ['ffmpeg', '-y', '-i', src_path, '-ac', '1', '-ar', str(sr), dst_path]
        subprocess.run(command, check=True)

# 📁 시각화 폴더 생성
for label_name in label_names:
    vis_path = os.path.join(output_dir, 'visuals_bview', label_name)
    os.makedirs(vis_path, exist_ok=True)

# 📦 전처리 시작
for label_name in label_names:
    folder_path = os.path.join(base_dir, label_name)
    label = label_map[label_name]

    if not os.path.exists(folder_path):
        print(f"⚠️ 폴더 없음: {folder_path}")
        continue

    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.wav', '.mp4', '.m4a', '.mp3'))]
    print(f"🎧 [{label_name}] 총 {len(files)}개 파일 처리 중...")

    for file_name in tqdm(files, desc=f"[{label_name}] 전처리", unit="file"):
        file_path = os.path.join(folder_path, file_name)

        # 🎧 mp4/m4a/mp3를 wav로 변환
        if file_name.lower().endswith(('.mp4', '.m4a', '.mp3')):
            wav_name = f"{label_name}_{os.path.splitext(file_name)[0]}.wav"
            wav_path = os.path.join(folder_path, wav_name)
            convert_to_wav(file_path, wav_path)
            file_path = wav_path

        base_filename = os.path.splitext(file_name)[0]

        # 📏 오디오 전체 길이 가져오기
        try:
            total_duration = librosa.get_duration(path=file_path)
        except:
            print(f"❗ duration 읽기 실패: {file_path}")
            continue

        # 📏 segment 개수 계산
        segment_count = int(total_duration // segment_duration)

        for i in range(segment_count):
            offset = i * segment_duration
            try:
                y_audio, _ = librosa.load(file_path, sr=sr, offset=offset, duration=segment_duration)
            except:
                print(f"❗ load 실패: {file_path} (segment {i})")
                continue

            # 📊 MFCC + ZCR 특징 추출
            mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
            zcr = librosa.feature.zero_crossing_rate(y=y_audio, hop_length=hop_length)
            features = np.vstack([mfcc, zcr])

            # ⛑️ 패딩 or 자르기 (max_len 맞추기)
            if features.shape[1] < max_len:
                features = np.pad(features, ((0, 0), (0, max_len - features.shape[1])), mode='constant')
            else:
                features = features[:, :max_len]

            # 💾 결과 저장
            X.append(features.T)
            y.append(label)

            # 🖼️ 스펙트럼 이미지 저장
            if save_visuals:
                save_path = os.path.join(output_dir, 'visuals_bview', label_name, f"{base_filename}_seg{i+1}.png")
                plt.figure(figsize=(10, 4))
                plt.imshow(features, aspect='auto', origin='lower', cmap='coolwarm')
                plt.title(f"{base_filename}_seg{i+1} - {label_name}")
                plt.xlabel("Frame")
                plt.ylabel("Feature Index (MFCC+ZCR)")
                plt.colorbar()
                plt.tight_layout()
                plt.savefig(save_path)
                plt.close()

# ✅ 전체 데이터 저장
os.makedirs(output_dir, exist_ok=True)
np.save(os.path.join(output_dir, "X_lstm_bview.npy"), np.array(X))
np.save(os.path.join(output_dir, "y_lstm_bview.npy"), np.array(y))

# ✅ 완료 출력
print("✅ segment_duration:", segment_duration)
print("✅ Calculated max_len:", max_len)
print("✅ X shape (LSTM용):", np.array(X).shape)
print("✅ y shape:", np.array(y).shape)
print("📁 저장 완료:", output_dir)
