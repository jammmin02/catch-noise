# 2-class 오디오 분류를 위한 LSTM 입력 데이터 전처리 스크립트
# 기능: 오디오 세그먼트 분할, MFCC+ZCR 추출, 시각화, 오버샘플링, 로그 및 요약 저장

import os
import librosa
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import pandas as pd
import random
import uuid
import json
import librosa.display
from tqdm import tqdm
from datetime import datetime
import argparse

# ---------------------- 파라미터 설정 ----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', type=str, default='data')  # 클래스별 원본 오디오 폴더가 위치한 디렉터리
parser.add_argument('--output_dir', type=str, default='dev/jungmin/2class_noisy_vs_nonnoisy/pyTorch_v2/outputs')  # 결과 저장 위치
parser.add_argument('--segment_duration', type=float, default=2.0)  # 세그먼트 단위 시간 (초)
parser.add_argument('--sr', type=int, default=44100)  # 샘플링 레이트 (USB 마이크와 동일하게 설정)
parser.add_argument('--n_mfcc', type=int, default=13)  # MFCC 추출 시 계수 수
parser.add_argument('--hop_length', type=int, default=512)  # 프레임 간 간격
parser.add_argument('--save_visuals', type=str, default='random', choices=['all', 'random', 'none'])  # 시각화 저장 옵션
parser.add_argument('--visual_prob', type=float, default=0.1)  # 랜덤 저장 확률
parser.add_argument('--sample_mode', action='store_true')  # 디버깅을 위한 샘플 모드 (2개 파일만 처리)
args = parser.parse_args([])  # Jupyter에서는 빈 리스트로 실행

# ---------------------- 라벨 자동 매핑 ----------------------
label_names = sorted([d for d in os.listdir(args.base_dir) if os.path.isdir(os.path.join(args.base_dir, d))])
label_map = {name: idx for idx, name in enumerate(label_names)}

# ---------------------- 프레임 길이 계산 ----------------------
frame_per_second = args.sr / args.hop_length  # 초당 프레임 수
max_len = int(frame_per_second * args.segment_duration)  # 세그먼트 당 프레임 수

# ---------------------- 결과 저장용 리스트 초기화 ----------------------
X, y, logs = [], [], []

# ---------------------- mp3 등 비 wav 파일 변환 ----------------------
def convert_to_wav(src_path, dst_path):
    if not os.path.exists(dst_path):
        command = ['ffmpeg', '-y', '-i', src_path, '-ac', '1', '-ar', str(args.sr), dst_path]
        subprocess.run(command, check=True)

# ---------------------- 클래스별 오디오 처리 ----------------------
for label_name in label_names:
    label = label_map[label_name]
    folder_path = os.path.join(args.base_dir, label_name)
    os.makedirs(os.path.join(args.output_dir, 'visuals', label_name), exist_ok=True)

    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.wav', '.mp3', '.m4a', '.mp4'))]
    if args.sample_mode:
        files = files[:2]  # 디버깅 모드일 경우 일부 파일만 처리

    for file_name in tqdm(files, desc=f"[{label_name}]"):
        file_path = os.path.join(folder_path, file_name)

        # 비 wav 파일 변환
        if file_name.endswith(('.mp3', '.m4a', '.mp4')):
            wav_path = os.path.splitext(file_path)[0] + '.wav'
            convert_to_wav(file_path, wav_path)
            file_path = wav_path

        try:
            total_duration = librosa.get_duration(path=file_path)
        except Exception as e:
            continue

        # 오디오를 세그먼트 단위로 분할
        for i in range(int(total_duration // args.segment_duration)):
            offset = i * args.segment_duration
            try:
                y_audio, _ = librosa.load(file_path, sr=args.sr, offset=offset, duration=args.segment_duration)
            except Exception as e:
                continue

            # 특징 추출 (MFCC + ZCR 평균)
            mfcc = librosa.feature.mfcc(y=y_audio, sr=args.sr, n_mfcc=args.n_mfcc, hop_length=args.hop_length)
            zcr = librosa.feature.zero_crossing_rate(y_audio, hop_length=args.hop_length)
            zcr_mean = np.mean(zcr)
            zcr_feature = np.full((1, mfcc.shape[1]), zcr_mean)  # MFCC와 같은 길이로 복제
            features = np.vstack([mfcc, zcr_feature])  # (14, T)

            # 패딩 또는 자르기 (세그먼트 길이 맞춤)
            if features.shape[1] < max_len:
                features = np.pad(features, ((0, 0), (0, max_len - features.shape[1])), mode='constant')
            else:
                features = features[:, :max_len]

            features_t = features.T.astype(np.float32)  # (T, 14)
            X.append(features_t)
            y.append(label)

            # 로그 기록
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

            # 시각화 저장 조건 검사
            save_flag = (
                args.save_visuals == 'all'
                or (args.save_visuals == 'random' and random.random() < args.visual_prob)
            )
            if save_flag:
                save_name = f"{label_name}_seg{i+1}_{log['uuid']}.png"
                save_path = os.path.join(args.output_dir, 'visuals', label_name, save_name)
                try:
                    fig, ax = plt.subplots(3, 1, figsize=(10, 8))
                    ax[0].plot(y_audio)
                    ax[0].set_title("Waveform")
                    librosa.display.specshow(mfcc, x_axis='time', sr=args.sr, hop_length=args.hop_length, ax=ax[1])
                    ax[1].set_title("MFCC")
                    ax[2].plot(zcr[0])
                    ax[2].set_title("ZCR")
                    plt.tight_layout()
                    plt.savefig(save_path)
                    plt.close()
                except Exception as e:
                    continue

# ---------------------- 클래스별 오버샘플링 처리 ----------------------
data_by_label = {i: [] for i in range(len(label_names))}
for f, l in zip(X, y):
    data_by_label[l].append(f)

mean_len = int(np.mean([len(v) for v in data_by_label.values() if v]))
X_balanced, y_balanced = [], []
for label, feats in data_by_label.items():
    if len(feats) >= mean_len:
        selected = random.sample(feats, mean_len)
    else:
        selected = feats * (mean_len // len(feats)) + random.sample(feats, mean_len % len(feats))
    X_balanced.extend(selected)
    y_balanced.extend([label] * mean_len)

# ---------------------- 결과 저장 ----------------------
os.makedirs(args.output_dir, exist_ok=True)
np.save(os.path.join(args.output_dir, "X_lstm.npy"), np.array(X_balanced))
np.save(os.path.join(args.output_dir, "y_lstm.npy"), np.array(y_balanced))
pd.DataFrame(logs).to_csv(os.path.join(args.output_dir, "segment_logs.csv"), index=False)

# ---------------------- 요약 정보 저장 ----------------------
summary = {
    'datetime': datetime.now().isoformat(),
    'segment_duration': args.segment_duration,
    'sample_rate': args.sr,
    'hop_length': args.hop_length,
    'n_mfcc': args.n_mfcc,
    'max_len': max_len,
    'label_map': label_map,
    'original_counts': {k: len(v) for k, v in data_by_label.items()},
    'oversampled_count': mean_len
}
with open(os.path.join(args.output_dir, "summary.json"), 'w') as f:
    json.dump(summary, f, indent=4)