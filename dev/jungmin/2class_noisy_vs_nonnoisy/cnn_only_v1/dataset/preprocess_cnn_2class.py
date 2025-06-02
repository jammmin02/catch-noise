import os
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import random
import uuid
import json
from tqdm import tqdm
from datetime import datetime
import argparse
from sklearn.preprocessing import StandardScaler
import joblib
import warnings

# 경고 메시지 억제 (불필요한 UserWarning 제거)
warnings.filterwarnings("ignore", category=UserWarning)

# 시드 고정 (재현성 확보)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ZCR 계산 함수 (torchaudio.functional 기반으로 통일)
def compute_zcr_torch(y_audio_tensor, hop_length):
    zcr = F.compute_zero_crossing_rate(
        y_audio_tensor.unsqueeze(0),
        frame_length=hop_length,
        hop_length=hop_length
    )
    return zcr.squeeze(0).numpy()

# 파라미터 설정 (커맨드라인 파라미터 혹은 기본값 사용)
parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', type=str, default=os.path.abspath('data'))
parser.add_argument('--output_dir', type=str, default=os.path.abspath('dev/jungmin/2class_noisy_vs_nonnoisy/cnn_only_v1/outputs'))
parser.add_argument('--segment_duration', type=float, default=2.0)
parser.add_argument('--sr', type=int, default=44100)
parser.add_argument('--n_mfcc', type=int, default=13)
parser.add_argument('--n_mels', type=int, default=40)
parser.add_argument('--hop_length', type=int, default=512)
parser.add_argument('--visuals_to_save', type=int, default=3)
parser.add_argument('--save_visuals', action='store_true')
parser.add_argument('--sample_size', type=int, default=None)
args = parser.parse_args([])

# 라벨 이름 자동 추출 (폴더명 기반)
label_names = sorted([d for d in os.listdir(args.base_dir) if os.path.isdir(os.path.join(args.base_dir, d))])
label_map = {name: idx for idx, name in enumerate(label_names)}

# 한 세그먼트당 프레임 수 계산
frame_per_second = args.sr / args.hop_length
max_len = int(np.ceil(frame_per_second * args.segment_duration))

# 데이터 저장을 위한 리스트 초기화
X, y, logs = [], [], []

# 전체 파일 리스트 준비
all_files = []
for label_name in label_names:
    folder_path = os.path.join(args.base_dir, label_name)
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.wav')]
    all_files.extend([(label_name, f) for f in files])

# 샘플 수 제한 (디버깅 등 빠른 테스트용)
if args.sample_size:
    all_files = all_files[:args.sample_size]

# 진행률 표시
pbar = tqdm(all_files, desc="Processing audio files")

# MFCC 추출기 초기화
mfcc_transform = T.MFCC(
    sample_rate=args.sr,
    n_mfcc=args.n_mfcc,
    melkwargs={'n_mels': args.n_mels, 'hop_length': args.hop_length}
)

# 메인 데이터 처리 루프
for label_name, file_path in all_files:
    label = label_map[label_name]
    os.makedirs(os.path.join(args.output_dir, 'visuals', label_name), exist_ok=True)
    file_name = os.path.basename(file_path)

    try:
        info = torchaudio.info(file_path)
        total_duration = info.num_frames / info.sample_rate
    except Exception as e:
        logs.append({'file': file_name, 'error': f"Failed to get duration - {e}"})
        pbar.update(1)
        continue

    for i in range(int(total_duration // args.segment_duration)):
        offset = int(i * args.segment_duration * args.sr)
        frames = int(args.segment_duration * args.sr)

        try:
            y_audio, sr = torchaudio.load(file_path, frame_offset=offset, num_frames=frames)
            y_audio = torch.mean(y_audio, dim=0)  # 스테레오 → 모노 변환
        except Exception as e:
            logs.append({'file': file_name, 'segment': i + 1, 'error': f"Load failed - {e}"})
            continue

        try:
            mfcc = mfcc_transform(y_audio.unsqueeze(0)).squeeze(0).numpy()  # MFCC 추출
            zcr = compute_zcr_torch(y_audio, hop_length=args.hop_length)    # ZCR 추출
        except Exception as e:
            logs.append({'file': file_name, 'segment': i + 1, 'error': f"Feature extraction failed - {e}"})
            continue

        zcr_mean = np.mean(zcr)
        zcr_feature = np.full((1, mfcc.shape[1]), zcr_mean)
        features = np.vstack([mfcc, zcr_feature])

        # 세그먼트 길이 맞추기 (padding or truncate)
        if features.shape[1] < max_len:
            pad_width = max_len - features.shape[1]
            features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant')
        else:
            features = features[:, :max_len]

        features_t = features.T.astype(np.float32)
        X.append(features_t)
        y.append(label)

        log = {
            'file': file_name,
            'segment': i + 1,
            'label': label_name,
            'uuid': str(uuid.uuid4()),
            'zcr_mean': float(zcr_mean),
            'mfcc_mean': float(np.mean(mfcc)),
            'features_shape': features_t.shape
        }
        logs.append(log)

    pbar.update(1)

pbar.close()

# Numpy 배열 변환 및 표준화 (scaler 저장)
X_array = np.array(X)
n_samples, time_steps, n_features = X_array.shape

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_array.reshape(-1, n_features)).reshape(n_samples, time_steps, n_features)
joblib.dump(scaler, os.path.join(args.output_dir, "scaler_cnn.pkl"))

# 클래스별 오버샘플링 수행 (라벨 밸런싱)
data_by_label = {i: [] for i in range(len(label_names))}
for f, l in zip(X_scaled, y):
    data_by_label[l].append(f)

lengths = [len(v) for v in data_by_label.values()]
mean_len = int(np.mean(lengths))

X_balanced, y_balanced = [], []
for label, feats in data_by_label.items():
    if len(feats) >= mean_len:
        selected = random.sample(feats, mean_len)
    else:
        selected = feats * (mean_len // len(feats)) + random.sample(feats, mean_len % len(feats))
    X_balanced.extend(selected)
    y_balanced.extend([label] * mean_len)

# 셔플 및 저장
combined = list(zip(X_balanced, y_balanced))
random.shuffle(combined)
X_balanced, y_balanced = zip(*combined)

os.makedirs(args.output_dir, exist_ok=True)
np.save(os.path.join(args.output_dir, "X_cnn.npy"), np.array(X_balanced))
np.save(os.path.join(args.output_dir, "y_cnn.npy"), np.array(y_balanced))
pd.DataFrame(logs).to_csv(os.path.join(args.output_dir, "segment_logs.csv"), index=False)

# 일부 샘플만 시각화 (전수 시각화 아님)
for idx in range(min(args.visuals_to_save, len(X))):
    sample = X[idx].T

    plt.figure(figsize=(10, 8))

    plt.subplot(3, 1, 1)
    plt.plot(sample[0])
    plt.ylim(-1, 1)
    plt.title("Waveform (Normalized)")

    plt.subplot(3, 1, 2)
    im = plt.imshow(sample[:-1], aspect='auto', origin='lower')
    plt.title("MFCC (Before Standardization)")
    plt.colorbar(im, format="%+2.0f")

    plt.subplot(3, 1, 3)
    plt.plot(sample[-1])
    plt.ylim(0, 0.5)
    plt.title("Zero Crossing Rate (ZCR)")

    plt.tight_layout()

    if args.save_visuals:
        plt.savefig(os.path.join(args.output_dir, f"visual_sample_{idx}.png"))
        plt.close()
    else:
        plt.show()
