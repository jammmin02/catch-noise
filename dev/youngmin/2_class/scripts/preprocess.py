import os
import shutil
import subprocess
import librosa
import numpy as np

# 1. mp3/m4a → wav 변환
def convert_to_wav(input_dir, output_dir, sr=22050):
    """
    지정된 폴더 내 mp3/m4a 파일을 wav로 변환하고,
    기존 wav 파일은 그대로 복사해서 output_dir로 이동
    """
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        ext = os.path.splitext(filename)[1].lower()
        output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".wav")

        if ext in [".mp3", ".m4a"]:
            # ffmpeg로 wav 변환
            subprocess.run(["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", str(sr), output_path],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif ext == ".wav":
            # 기존 wav 파일은 복사
            shutil.copy(input_path, output_path)

# 2. MFCC + ZCR 특징 추출
def extract_features(y, sr=22050, n_mfcc=13, hop_length=512):
    """
    입력된 오디오 y에서 MFCC 13개 + ZCR 1개 추출하여 (T, 14) 형태로 반환
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)
    return np.vstack([mfcc, zcr]).T  # (T, 14)

def process_all_audio(base_path, segment_duration=2.0, sr=22050):
    X_by_class = {0: [], 1: []}
    max_len = int((sr / 512) * segment_duration)

    for label, cls in enumerate(['non_noisy', 'noisy']):
        raw_path = os.path.join(base_path, "raw", cls)
        wav_path = os.path.join(base_path, "temp_wav", cls)
        convert_to_wav(raw_path, wav_path)

        for fname in os.listdir(wav_path):
            if not fname.endswith(".wav"):
                continue
            fpath = os.path.join(wav_path, fname)
            try:
                y_raw, _ = librosa.load(fpath, sr=sr)
            except Exception as e:
                print(f"Failed to load {fname}: {e}")
                continue

            seg_len = int(sr * segment_duration)
            for i in range(0, len(y_raw), seg_len):
                segment = y_raw[i:i + seg_len]
                if len(segment) < seg_len:
                    continue

                feat = extract_features(segment, sr=sr)
                feat = np.pad(feat, ((0, max_len - feat.shape[0]), (0, 0)), mode='constant') \
                       if feat.shape[0] < max_len else feat[:max_len]

                X_by_class[label].append(feat)

    # 오버샘플링 적용
    len_0 = len(X_by_class[0])
    len_1 = len(X_by_class[1])
    if len_0 > len_1:
        reps = (len_0 // len_1) + 1
        X_by_class[1] = (X_by_class[1] * reps)[:len_0]
    elif len_1 > len_0:
        reps = (len_1 // len_0) + 1
        X_by_class[0] = (X_by_class[0] * reps)[:len_1]

    # 병합
    X = np.array(X_by_class[0] + X_by_class[1])
    y = np.array([0] * len(X_by_class[0]) + [1] * len(X_by_class[1]))

    # 셔플
    indices = np.random.permutation(len(y))
    X = X[indices]
    y = y[indices]

    # 저장
    save_path = os.path.join(base_path, "processed")
    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, "x.npy"), X)
    np.save(os.path.join(save_path, "y.npy"), y)

    return X, y

# 4. 메인 실행 블록
if __name__ == "__main__":
    base_dir = "data"  # 루트 디렉토리 기준
    print("Starting preprocessing...")
    X, y = process_all_audio(base_dir)
    print(f"Done. X shape: {X.shape}, y shape: {y.shape}")
