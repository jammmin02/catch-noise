import os
import librosa
import soundfile as sf
import hashlib
import pandas as pd
from tqdm import tqdm
import mlflow

# 디렉토리 및 설정
DATA_DIR = "./data"
OUTPUT_DIR = "./dataset/raw_segments"
LABEL_CSV = "./dataset/labels.csv"
SEGMENT_DURATION = 1.0  # segment 길이 (초)
SLIDE_DURATION = 0.5    # 슬라이딩 간격 (초)
SR = 16000              # 샘플링 레이트 (Hz)

# 출력 디렉토리 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 라벨 정보를 담을 리스트
labels = []

def compute_file_hash(file_path):
    """
    파일의 SHA256 해시를 계산 (데이터셋 무결성 확인용)
    """
    BUF_SIZE = 65536
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(BUF_SIZE):
            sha256.update(chunk)
    return sha256.hexdigest()

def process_audio(file_path, label_name):
    """
    오디오 파일을 segment로 분할하고 저장 + 라벨 기록
    """
    y, _ = librosa.load(file_path, sr=SR)
    file_base = os.path.splitext(os.path.basename(file_path))[0]
    total_duration = librosa.get_duration(y=y, sr=SR)

    seg_idx = 0
    # segment 단위로 슬라이딩하며 처리
    for start in tqdm(range(0, int((total_duration - SEGMENT_DURATION) * SR + 1), int(SLIDE_DURATION * SR))):
        end = start + int(SEGMENT_DURATION * SR)
        segment = y[start:end]

        # segment 길이가 부족하면 zero-padding
        if len(segment) < int(SEGMENT_DURATION * SR):
            segment = librosa.util.fix_length(segment, int(SEGMENT_DURATION * SR))

        segment_name = f"{file_base}_seg{seg_idx:03d}.wav"
        out_path = os.path.join(OUTPUT_DIR, segment_name)

        # segment 파일 저장
        sf.write(out_path, segment, SR)

        # 라벨 정보 기록
        labels.append({
            "filename": segment_name,
            "label": label_name
        })

        seg_idx += 1

def main():
    """
    전체 데이터 디렉토리 순회하며 segment 생성 및 라벨링
    """
    for label_dir in os.listdir(DATA_DIR):
        class_dir = os.path.join(DATA_DIR, label_dir)
        if not os.path.isdir(class_dir):
            continue

        # 클래스 디렉토리 내 .wav 파일 순회
        for file in os.listdir(class_dir):
            if not file.endswith(".wav"):
                continue
            file_path = os.path.join(class_dir, file)
            process_audio(file_path, label_dir)

    # 라벨 CSV 파일 저장
    df = pd.DataFrame(labels)
    df.to_csv(LABEL_CSV, index=False)
    print(f"Saved labels: {LABEL_CSV}")

    # 데이터셋 파일 해시 계산 및 통합 해시 생성
    hash_list = []
    for root, _, files in os.walk(OUTPUT_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            file_hash = compute_file_hash(file_path)
            hash_list.append(file_hash)

    dataset_hash = hashlib.sha256("".join(sorted(hash_list)).encode()).hexdigest()
    print(f"Dataset hash: {dataset_hash}")

    # MLflow에 파라미터 및 해시 기록
    mlflow.set_tracking_uri("http://mlflow:5000")  # 필요 시 URI 수정
    with mlflow.start_run(run_name="dataset_preparation"):
        mlflow.log_param("segment_duration", SEGMENT_DURATION)
        mlflow.log_param("slide_duration", SLIDE_DURATION)
        mlflow.log_param("sample_rate", SR)
        mlflow.log_param("dataset_hash", dataset_hash)

if __name__ == "__main__":
    main()