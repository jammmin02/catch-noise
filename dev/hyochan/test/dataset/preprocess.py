import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# ==============================
# 경로 설정 (상단 고정)
INPUT_DIR = "hyochan/test/data"                # 입력 폴더
OUTPUT_DIR = "dataset/output"     # 출력 폴더
# ==============================

# 전처리 파라미터 설정
SAMPLE_RATE = 22050
N_MFCC = 13
SEGMENT_DURATION = 2.0  # 초
HOP_LENGTH = 512

def save_mfcc_image(mfcc, save_path):
    plt.figure(figsize=(6, 3))
    librosa.display.specshow(mfcc, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def extract_features(audio_path, label, file_id, image_output_dir):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    segment_length = int(SEGMENT_DURATION * sr)

    features = []
    energy_list = []

    for idx, start in enumerate(range(0, len(y), segment_length)):
        end = start + segment_length
        if end > len(y):
            break

        segment = y[start:end]
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
        zcr = librosa.feature.zero_crossing_rate(y=segment, hop_length=HOP_LENGTH)
        energy = np.mean(librosa.feature.rms(y=segment))

        feature = np.vstack([mfcc, zcr]).T
        features.append(feature)
        energy_list.append(energy)

        # 이미지 저장
        save_dir = os.path.join(image_output_dir, label)
        os.makedirs(save_dir, exist_ok=True)
        image_path = os.path.join(save_dir, f"{file_id}_seg{idx}.png")
        save_mfcc_image(mfcc, image_path)

    return np.array(features), np.array(energy_list)

def preprocess_dataset(input_dir, output_dir):
    all_X = []
    all_E = []
    all_y = []

    image_dir = os.path.join(output_dir, "mfcc_images")
    os.makedirs(image_dir, exist_ok=True)

    for label in os.listdir(input_dir):
        class_dir = os.path.join(input_dir, label)
        if not os.path.isdir(class_dir):
            continue

        for fname in os.listdir(class_dir):
            if not fname.endswith(".wav"):
                continue
            path = os.path.join(class_dir, fname)
            file_id = os.path.splitext(fname)[0]

            try:
                feats, energies = extract_features(path, label, file_id, image_dir)
                all_X.append(feats)
                all_E.append(energies)
                all_y.extend([label] * feats.shape[0])
                print(f"[✓] {fname} → {feats.shape[0]} segments & MFCC 이미지 저장됨")
            except Exception as e:
                print(f"[ERROR] {fname}: {e}")

    # 저장
    X = np.concatenate(all_X, axis=0)
    E = np.concatenate(all_E, axis=0)
    y = np.array(all_y)

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "X.npy"), X)
    np.save(os.path.join(output_dir, "E.npy"), E)
    np.save(os.path.join(output_dir, "y.npy"), y)

    print(f"\n전처리 완료: 총 {len(y)} 세그먼트")
    print(f" - X shape: {X.shape}")
    print(f" - E shape: {E.shape}")
    print(f" - y shape: {y.shape}")
    print(f" - MFCC 이미지 저장 위치: {image_dir}/[클래스명]/*.png")

# 실행
if __name__ == "__main__":
    preprocess_dataset(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR)
