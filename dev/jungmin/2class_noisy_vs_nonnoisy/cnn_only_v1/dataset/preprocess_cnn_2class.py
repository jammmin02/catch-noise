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
from datetime import datetime
import argparse
from sklearn.preprocessing import StandardScaler
import joblib
import multiprocessing as mp
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ZCR 계산 (torchaudio 기반)
def compute_zcr_torch(y_audio_tensor, hop_length):
    zcr = F.compute_zero_crossing_rate(
        y_audio_tensor.unsqueeze(0),
        frame_length=hop_length,
        hop_length=hop_length
    )
    return zcr.squeeze(0).numpy()

# 전역 파라미터
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
parser.add_argument('--num_workers', type=int, default=mp.cpu_count())
args = parser.parse_args([])

label_names = sorted([d for d in os.listdir(args.base_dir) if os.path.isdir(os.path.join(args.base_dir, d))])
label_map = {name: idx for idx, name in enumerate(label_names)}

frame_per_second = args.sr / args.hop_length
max_len = int(np.ceil(frame_per_second * args.segment_duration))

# torchaudio transform 초기화 (worker 단위)
def init_worker_transform():
    global mfcc_transform
    mfcc_transform = T.MFCC(
        sample_rate=args.sr,
        n_mfcc=args.n_mfcc,
        melkwargs={'n_mels': args.n_mels, 'hop_length': args.hop_length}
    )

# 각 파일 단위 전처리 함수 (worker 단위 병렬화)
def process_file(item):
    label_name, file_path = item
    label = label_map[label_name]
    file_name = os.path.basename(file_path)
    logs = []
    segments = []

    try:
        info = torchaudio.info(file_path)
        total_duration = info.num_frames / info.sample_rate
    except Exception as e:
        logs.append({'file': file_name, 'error': str(e)})
        return segments, logs

    for i in range(int(total_duration // args.segment_duration)):
        offset = int(i * args.segment_duration * args.sr)
        frames = int(args.segment_duration * args.sr)

        try:
            y_audio, sr = torchaudio.load(file_path, frame_offset=offset, num_frames=frames)
            y_audio = torch.mean(y_audio, dim=0)

            mfcc = mfcc_transform(y_audio.unsqueeze(0)).squeeze(0).numpy()
            zcr = compute_zcr_torch(y_audio, hop_length=args.hop_length)

            zcr_mean = np.mean(zcr)
            zcr_feature = np.full((1, mfcc.shape[1]), zcr_mean)
            features = np.vstack([mfcc, zcr_feature])

            if features.shape[1] < max_len:
                pad_width = max_len - features.shape[1]
                features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant')
            else:
                features = features[:, :max_len]

            features_t = features.T.astype(np.float32)
            segments.append((features_t, label))

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

        except Exception as e:
            logs.append({'file': file_name, 'segment': i + 1, 'error': str(e)})

    return segments, logs

# 전체 파일 리스트 구성
all_files = []
for label_name in label_names:
    folder_path = os.path.join(args.base_dir, label_name)
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.wav')]
    all_files.extend([(label_name, f) for f in files])

# 샘플 수 제한
if args.sample_size:
    all_files = all_files[:args.sample_size]

total_files = len(all_files)

# 실시간 진행률 표시용 공유 변수
from multiprocessing import Value, Lock

counter = Value('i', 0)
lock = Lock()

# 진행률 출력 래퍼 함수
def wrapped_process_file(item):
    segments, logs = process_file(item)
    with lock:
        counter.value += 1
        current = counter.value
        ratio = current / total_files
        bar_len = 30
        done_len = int(ratio * bar_len)
        bar = "[" + "=" * done_len + ">" + " " * (bar_len - done_len - 1) + "]"
        print(f"\r진행률 {bar} {ratio*100:.1f}% ({current}/{total_files})", end="", flush=True)
    return segments, logs

# 병렬 처리 시작
print(f"총 {total_files}개 파일 처리 시작 (병렬 workers: {args.num_workers})")

X, y, logs_all = [], [], []

with mp.Pool(processes=args.num_workers, initializer=init_worker_transform) as pool:
    for segments, logs in pool.imap_unordered(wrapped_process_file, all_files):
        for features_t, label in segments:
            X.append(features_t)
            y.append(label)
        logs_all.extend(logs)

print("\n전처리 완료 (병렬 최적화 + 실시간 시각화 적용)!")

# numpy 변환 및 표준화
X_array = np.array(X)
n_samples, time_steps, n_features = X_array.shape

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_array.reshape(-1, n_features)).reshape(n_samples, time_steps, n_features)
joblib.dump(scaler, os.path.join(args.output_dir, "scaler_cnn.pkl"))

# 오버샘플링 (라벨 균형 유지)
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
pd.DataFrame(logs_all).to_csv(os.path.join(args.output_dir, "segment_logs.csv"), index=False)

# 일부 샘플만 시각화
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

print("전처리 및 시각화까지 전체 완료!")