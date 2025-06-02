import os
import numpy as np
import soundfile as sf
import subprocess
import matplotlib.pyplot as plt
import pandas as pd
import random
import uuid
import json
import torch
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm
from datetime import datetime
import argparse

# ---------------------- 파라미터 설정 ----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', type=str, default=os.path.abspath('data'))
parser.add_argument('--output_dir', type=str, default=os.path.abspath('dev/jungmin/2class_noisy_vs_nonnoisy/cnn_only_v2/outputs'))
parser.add_argument('--segment_duration', type=float, default=2.0)
parser.add_argument('--sr', type=int, default=44100)
parser.add_argument('--n_mfcc', type=int, default=13)
parser.add_argument('--hop_length', type=int, default=512)
parser.add_argument('--save_visuals', type=str, default='random', choices=['all', 'random', 'none'])
parser.add_argument('--visual_prob', type=float, default=0.1)
parser.add_argument('--sample_mode', action='store_true')
args = parser.parse_args([])

# ---------------------- 라벨 매핑 ----------------------
label_names = sorted([d for d in os.listdir(args.base_dir) if os.path.isdir(os.path.join(args.base_dir, d))])
label_map = {name: idx for idx, name in enumerate(label_names)}

# ---------------------- 길이 계산 ----------------------
frame_per_second = args.sr / args.hop_length
max_len = int(np.ceil(frame_per_second * args.segment_duration))

# ---------------------- MFCC 변환기 ----------------------
mfcc_transform = T.MFCC(
    sample_rate=args.sr,
    n_mfcc=args.n_mfcc,
    melkwargs={"n_fft": 2048, "hop_length": args.hop_length}
)

# ---------------------- 결과 저장용 리스트 ----------------------
X, y, logs = [], [], []

# ---------------------- 비 wav 변환 ----------------------
def convert_to_wav(src_path, dst_path):
    if not os.path.exists(dst_path):
        command = ['ffmpeg', '-y', '-i', src_path, '-ac', '1', '-ar', str(args.sr), dst_path]
        subprocess.run(command, check=True)

# ---------------------- 클래스별 처리 ----------------------
for label_name in label_names:
    label = label_map[label_name]
    folder_path = os.path.join(args.base_dir, label_name)
    os.makedirs(os.path.join(args.output_dir, 'visuals', label_name), exist_ok=True)

    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.wav', '.mp3', '.m4a', '.mp4'))]
    if args.sample_mode:
        files = files[:2]

    for file_name in tqdm(files, desc=f"[{label_name}]"):
        file_path = os.path.join(folder_path, file_name)

        if file_name.endswith(('.mp3', '.m4a', '.mp4')):
            wav_path = os.path.splitext(file_path)[0] + '.wav'
            convert_to_wav(file_path, wav_path)
            file_path = wav_path

        try:
            info = sf.info(file_path)
            total_duration = info.frames / info.samplerate
        except Exception as e:
            print(f"[길이 확인 실패] {file_path} - {e}")
            continue

        for i in range(int(total_duration // args.segment_duration)):
            offset = int(i * args.segment_duration * args.sr)
            frames = int(args.segment_duration * args.sr)

            # 🛑 읽을 샘플 부족 방지
            if offset + frames > info.frames:
                continue

            try:
                y_audio, _ = sf.read(file_path, start=offset, frames=frames)
                if y_audio.ndim > 1:
                    y_audio = np.mean(y_audio, axis=1)
                if len(y_audio) == 0:
                    print(f"[경고] 비어있는 segment 건너뜀: {file_path}, segment {i+1}")
                    continue
            except Exception as e:
                print(f"[로드 실패] {file_path}, offset={offset} - {e}")
                continue

            try:
                # torchaudio → batch 차원 추가
                audio_tensor = torch.tensor(y_audio, dtype=torch.float32).unsqueeze(0)
                mfcc = mfcc_transform(audio_tensor).squeeze(0).numpy()  # (n_mfcc, time_steps)

                # padding
                if mfcc.shape[1] < max_len:
                    pad_width = max_len - mfcc.shape[1]
                    mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
                elif mfcc.shape[1] > max_len:
                    mfcc = mfcc[:, :max_len]

                # transpose to (time_steps, n_mfcc)
                features_t = mfcc.T.astype(np.float32)
                X.append(features_t)
                y.append(label)

                log = {
                    'file': file_name,
                    'segment': i + 1,
                    'label': label_name,
                    'uuid': str(uuid.uuid4()),
                    'mfcc_mean': float(np.mean(mfcc)),
                    'features_shape': features_t.shape
                }
                logs.append(log)

                # ✅ 시각화 (확률 기반으로 일부만 저장)
                save_flag = (
                    args.save_visuals == 'all'
                    or (args.save_visuals == 'random' and random.random() < args.visual_prob)
                )

                if save_flag:
                    save_name = f"{label_name}_seg{i+1}_{log['uuid']}.png"
                    save_path = os.path.join(args.output_dir, 'visuals', label_name, save_name)

                    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
                    ax[0].plot(y_audio)
                    ax[0].set_title("Waveform")

                    img = ax[1].imshow(mfcc, origin='lower', aspect='auto', cmap='coolwarm',
                                       extent=[0, args.segment_duration, 0, args.n_mfcc])
                    ax[1].set_title("MFCC")
                    ax[1].set_xlabel("Time")
                    ax[1].set_ylabel("MFCC Coefficients")

                    plt.tight_layout()
                    plt.savefig(save_path)
                    plt.close()

            except Exception as e:
                print(f"[특징 추출 실패] {file_path}, segment={i+1} - {e}")
                continue

# ---------------------- 클래스별 오버샘플링 ----------------------
data_by_label = {i: [] for i in range(len(label_names))}
for f, l in zip(X, y):
    data_by_label[l].append(f)

lengths = [len(v) for v in data_by_label.values() if len(v) > 0]
if not lengths:
    raise ValueError("데이터 수집 실패: 세그먼트 없음")

mean_len = int(np.mean(lengths))

X_balanced, y_balanced = [], []
for label, feats in data_by_label.items():
    if len(feats) >= mean_len:
        selected = random.sample(feats, mean_len)
    else:
        selected = feats * (mean_len // len(feats)) + random.sample(feats, mean_len % len(feats))
    X_balanced.extend(selected)
    y_balanced.extend([label] * mean_len)

# ---------------------- 셔플 ----------------------
combined = list(zip(X_balanced, y_balanced))
random.shuffle(combined)
X_balanced, y_balanced = zip(*combined)

# ---------------------- 저장 ----------------------
os.makedirs(args.output_dir, exist_ok=True)
np.save(os.path.join(args.output_dir, "X_cnn.npy"), np.array(X_balanced))
np.save(os.path.join(args.output_dir, "y_cnn.npy"), np.array(y_balanced))
pd.DataFrame(logs).to_csv(os.path.join(args.output_dir, "segment_logs.csv"), index=False)

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
