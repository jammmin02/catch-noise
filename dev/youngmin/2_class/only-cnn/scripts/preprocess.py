import os
import numpy as np
import librosa
from tqdm import tqdm
from pathlib import Path
import soundfile as sf

# ============================================
# 설정: 경로 및 파라미터
# ============================================

# 원본 오디오가 있는 경로: class별 하위 폴더 필요 (non_noisy, noisy)
RAW_DIR = Path("/workspace/data/raw")

# 전처리 결과 저장 위치
SAVE_X = Path("/workspace/data/x.npy")
SAVE_Y = Path("/workspace/data/y.npy")

# 오디오 처리 파라미터
SR = 22050                # 샘플링 레이트 (Hz)
N_MFCC = 13               # MFCC 계수 수
SEGMENT_DURATION = 2.0    # 한 segment 길이 (초)
HOP_LENGTH = 512          # MFCC 추출 시 hop 간격 (샘플 단위)

# ============================================
# 함수: 오디오에서 MFCC segment 추출
# ============================================

def extract_mfcc_segments(file_path, sr=SR, n_mfcc=N_MFCC, segment_duration=2.0):
    """
    하나의 오디오 파일을 읽고, segment_duration 길이로 나누어
    각 segment에 대해 MFCC를 추출하여 리스트로 반환.
    """
    y, _ = librosa.load(file_path, sr=sr)
    segment_len = int(sr * segment_duration)  # segment 길이 (샘플 수)

    mfcc_segments = []

    # segment 단위로 오디오를 잘라서 처리
    for start in range(0, len(y) - segment_len + 1, segment_len):
        segment = y[start:start + segment_len]

        # MFCC 추출: (n_mfcc, time_frame)
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc, hop_length=HOP_LENGTH)
        mfcc = mfcc.T  # transpose → (time_frame, n_mfcc)

        # CNN에 맞는 입력 길이로 고정
        if mfcc.shape[0] >= 86:
            mfcc_segments.append(mfcc[:86])  # 2초 기준 86 프레임 확보

    return mfcc_segments

# ============================================
# 함수: 전체 데이터 로딩 및 라벨링
# ============================================

def load_data(raw_dir):
    """
    주어진 raw 디렉토리 내에서 클래스 폴더를 순회하며
    MFCC segment를 추출하고, 라벨링하여 X, y 리스트 생성.
    """
    x_list = []
    y_list = []

    # 클래스명 → 정수 라벨 매핑
    label_map = {"non_noisy": 0, "noisy": 1}

    # 클래스별 폴더 순회
    for label_name, label_idx in label_map.items():
        folder = raw_dir / label_name
        if not folder.exists():
            continue

        # 지원되는 모든 확장자 처리
        files = list(folder.glob("*.*"))

        for file in tqdm(files, desc=f"Loading {label_name}"):
            try:
                segments = extract_mfcc_segments(file)
                for seg in segments:
                    x_list.append(seg)
                    y_list.append(label_idx)
            except Exception as e:
                print(f"[ERROR] {file}: {e}")

    return np.array(x_list), np.array(y_list)

# ============================================
# 메인: 전처리 실행 및 저장
# ============================================

if __name__ == "__main__":
    print("Starting MFCC-only 2초 분할 전처리...")

    # 전체 데이터 로드
    x, y = load_data(RAW_DIR)

    # CNN 입력 형태로 reshape → (N, 86, 13, 1)
    x = x[..., np.newaxis]

    # 결과 출력 및 저장
    print(f"Done. Shape X: {x.shape}, Y: {y.shape}")
    np.save(SAVE_X, x)
    np.save(SAVE_Y, y)
