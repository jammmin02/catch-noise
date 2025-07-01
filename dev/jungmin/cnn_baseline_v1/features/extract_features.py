import os
import numpy as np
import librosa
from tqdm import tqdm
import hashlib
import mlflow

# 입력 및 출력 디렉토리 설정
INPUT_DIR = "./dataset/raw_segments"
OUTPUT_DIR = "./dataset/processed"
SR = 16000      # 샘플링 레이트 (Hz)
N_MFCC = 13     # MFCC 개수

# 출력 디렉토리 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

def zscore(x):
    """
    Z-score 정규화 (평균 0, 표준편차 1)
    std=0 방지를 위해 작은 값 1e-9 더함
    """
    mean = np.mean(x)
    std = np.std(x) + 1e-9
    return (x - mean) / std

def minmax(x):
    """
    Min-Max 정규화 (0~1 범위)
    max == min일 경우 0 배열 반환
    """
    min_val = np.min(x)
    max_val = np.max(x)
    if max_val - min_val < 1e-9:
        return np.zeros_like(x)
    return (x - min_val) / (max_val - min_val)

def process_file(file_path):
    """
    오디오 파일로부터 MFCC, ZCR, RMS 추출 및 정규화
    - MFCC: Z-score + Min-Max
    - ZCR: 0~1 그대로
    - RMS: log + Min-Max
    """
    y, _ = librosa.load(file_path, sr=SR)

    # MFCC 추출 및 정규화
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
    mfcc = zscore(mfcc)
    mfcc = minmax(mfcc)

    # ZCR 추출 (0~1 범위)
    zcr = librosa.feature.zero_crossing_rate(y)

    # RMS 추출 및 log + Min-Max 정규화
    rms = librosa.feature.rms(y=y)
    rms = np.log10(rms + 1e-9)
    rms = minmax(rms)

    # 모든 특징을 수직 방향으로 병합
    combined = np.vstack([mfcc, zcr, rms])

    return combined

def compute_dataset_hash(folder):
    """
    .npy 파일들의 해시를 통합하여 데이터셋 전체 해시 생성
    """
    hash_list = []
    for file in os.listdir(folder):
        if file.endswith(".npy"):
            path = os.path.join(folder, file)
            data = np.load(path)
            # 파일 데이터에 대한 SHA256 해시
            hash_list.append(hashlib.sha256(data.tobytes()).hexdigest())
    # 개별 해시를 정렬 후 통합해 최종 해시 생성
    dataset_hash = hashlib.sha256("".join(sorted(hash_list)).encode()).hexdigest()
    return dataset_hash

def main():
    """
    전체 오디오 segment 파일 처리:
    - 특징 추출 및 정규화
    - .npy 파일 저장
    - 데이터셋 hash 계산
    - MLflow에 기록
    """
    mlflow.set_tracking_uri("http://mlflow:5000")  # MLflow 서버 URI (필요 시 변경)
    with mlflow.start_run(run_name="feature_extraction"):
        # 전처리 파라미터 기록
        mlflow.log_param("sample_rate", SR)
        mlflow.log_param("n_mfcc", N_MFCC)
        mlflow.log_param("preprocessing_mfcc", "zscore+minmax")
        mlflow.log_param("preprocessing_rms", "log+minmax")
        mlflow.log_param("preprocessing_zcr", "0-1 그대로")

        # 각 .wav 파일 처리
        for file in tqdm(os.listdir(INPUT_DIR)):
            if not file.endswith(".wav"):
                continue
            file_path = os.path.join(INPUT_DIR, file)
            features = process_file(file_path)

            # .npy 파일로 저장
            out_path = os.path.join(OUTPUT_DIR, file.replace(".wav", ".npy"))
            np.save(out_path, features)

        # 데이터셋 해시 계산 및 기록
        dataset_hash = compute_dataset_hash(OUTPUT_DIR)
        mlflow.log_param("dataset_hash", dataset_hash)

        # 완료 메시지 출력
        print(f"Features saved in {OUTPUT_DIR}")
        print(f"Dataset hash: {dataset_hash}")

if __name__ == "__main__":
    main()
