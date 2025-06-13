import os
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import pandas as pd
import random
import uuid
import argparse
import concurrent.futures as cf
import json
from datetime import datetime
import warnings
from multiprocessing import Value, Lock

warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

CODE_VERSION = "robust_v7_torchaudio_numpy_zcr_with_visual"
AUGMENTATION_FACTOR = 2
VISUAL_SAMPLE_PROB = 0.1  # 시각화 샘플링 확률

# numpy 기반 ZCR
def compute_zcr(y_audio_tensor, frame_size, hop_size):
    y_np = y_audio_tensor.numpy()
    zcr = []
    for i in range(0, len(y_np) - frame_size, hop_size):
        frame = y_np[i:i+frame_size]
        crossings = np.diff(np.sign(frame))
        zcr_val = (crossings != 0).sum() / frame_size
        zcr.append(zcr_val)
    return np.array(zcr)

# 데이터 증강 (time shift)
def augment_time_shift(y, max_shift=0.2):
    shift_amt = int(random.uniform(-max_shift, max_shift) * len(y))
    return torch.roll(y, shifts=shift_amt, dims=0)

parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', type=str, default=os.path.abspath('data'))
parser.add_argument('--output_dir', type=str, default=os.path.abspath('dev/jungmin/2class_noisy_vs_nonnoisy/cnn_only_v1/outputs'))
parser.add_argument('--segment_duration', type=float, default=2.0)
parser.add_argument('--target_sr', type=int, default=44100)
parser.add_argument('--n_mfcc', type=int, default=13)
parser.add_argument('--n_mels', type=int, default=40)
parser.add_argument('--hop_length', type=int, default=512)
parser.add_argument('--n_fft', type=int, default=2048)
parser.add_argument('--num_workers', type=int, default=4)
args = parser.parse_args([])

label_names = ['non_noisy', 'noisy']
label_map = {name: idx for idx, name in enumerate(label_names)}

frame_per_second = args.target_sr / args.hop_length
max_len = int(np.ceil(frame_per_second * args.segment_duration))

all_files = []
for label_name in label_names:
    folder_path = os.path.join(args.base_dir, label_name)
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            if f.lower().endswith('.wav'):
                full_path = os.path.abspath(os.path.join(root, f))
                all_files.append((label_name, full_path))

total_files = len(all_files)
num_workers = min(args.num_workers, total_files)
counter = Value('i', 0)
lock = Lock()

def load_audio(file_path, target_sr):
    y_audio, sr = torchaudio.load(file_path)
    y_audio = torch.mean(y_audio, dim=0)
    if sr != target_sr:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        y_audio = resampler(y_audio)
    return y_audio

def extract_features(y_audio_tensor, label, segment_idx, logs, save_dir, label_name, augment=False):
    segments = []
    try:
        if augment:
            y_audio_tensor = augment_time_shift(y_audio_tensor)

        mfcc_transform = T.MFCC(
            sample_rate=args.target_sr,
            n_mfcc=args.n_mfcc,
            melkwargs={
                'n_fft': args.n_fft,
                'hop_length': args.hop_length,
                'n_mels': args.n_mels
            }
        )

        mfcc = mfcc_transform(y_audio_tensor.unsqueeze(0)).squeeze(0).numpy()

        if mfcc.shape[1] == 0:
            logs.append({'segment': segment_idx, 'error': 'Empty MFCC'})
            return []

        zcr = compute_zcr(y_audio_tensor, frame_size=args.n_fft, hop_size=args.hop_length)
        zcr_mean = np.mean(zcr)
        zcr_feature = np.full((1, mfcc.shape[1]), zcr_mean)
        features = np.vstack([mfcc, zcr_feature])

        total_pad = max_len - features.shape[1]
        if total_pad > 0:
            left_pad = total_pad // 2
            right_pad = total_pad - left_pad
            features = np.pad(features, ((0, 0), (left_pad, right_pad)), mode='constant')
        else:
            features = features[:, :max_len]

        uuid_name = f"{uuid.uuid4()}.npy"
        np.save(os.path.join(save_dir, uuid_name), features.astype(np.float32))
        segments.append(uuid_name)

        logs.append({
            'segment': segment_idx, 'label': label, 'uuid': uuid_name,
            'zcr_mean': float(zcr_mean), 'mfcc_mean': float(np.mean(mfcc)), 'shape': features.shape
        })

        if random.random() < VISUAL_SAMPLE_PROB:
            save_visualization(y_audio_tensor.numpy(), mfcc, zcr, label_name, uuid_name)

    except Exception as e:
        logs.append({'segment': segment_idx, 'error': str(e)})
    
    return segments

def save_visualization(y_audio, mfcc, zcr, label_name, uuid_name):
    vis_dir = os.path.join(args.output_dir, "visuals", label_name)
    os.makedirs(vis_dir, exist_ok=True)
    plt.figure(figsize=(10, 8))

    plt.subplot(3, 1, 1)
    plt.plot(y_audio)
    plt.title("Waveform")

    plt.subplot(3, 1, 2)
    plt.imshow(mfcc, aspect='auto', origin='lower')
    plt.title("MFCC")
    plt.colorbar()

    plt.subplot(3, 1, 3)
    plt.plot(zcr)
    plt.title("Zero Crossing Rate (ZCR)")

    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f"{uuid_name}.png"))
    plt.close()

def process_file(item):
    label_name, file_path = item
    label = label_map[label_name]
    file_name = os.path.basename(file_path)
    logs = []
    segments = []

    try:
        y_audio = load_audio(file_path, args.target_sr)
        total_samples = len(y_audio)
        samples_per_segment = int(args.segment_duration * args.target_sr)
        n_segments = total_samples // samples_per_segment

        if n_segments == 0:
            pad_len = samples_per_segment - total_samples
            if pad_len < 0:
                logs.append({'file': file_name, 'error': 'Negative pad length'})
                return [], logs
            y_audio = torch.nn.functional.pad(y_audio, (0, pad_len))
            segments.extend(extract_features(y_audio, label, 1, logs, segment_dir, label_name))
            for _ in range(AUGMENTATION_FACTOR):
                segments.extend(extract_features(y_audio, label, 1, logs, segment_dir, label_name, augment=True))
        else:
            for i in range(n_segments):
                start = i * samples_per_segment
                end = start + samples_per_segment
                segment = y_audio[start:end]
                segments.extend(extract_features(segment, label, i+1, logs, segment_dir, label_name))
                for _ in range(AUGMENTATION_FACTOR):
                    segments.extend(extract_features(segment, label, i+1, logs, segment_dir, label_name, augment=True))

    except Exception as e:
        logs.append({'file': file_name, 'error': str(e)})

    return [(s, label) for s in segments], logs

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

# 디렉토리 준비
os.makedirs(args.output_dir, exist_ok=True)
segment_dir = os.path.join(args.output_dir, "segments")
os.makedirs(segment_dir, exist_ok=True)

X_info, logs_all = [], []

print(f"총 {total_files}개 파일 처리 시작 (병렬 workers: {num_workers})")

with cf.ProcessPoolExecutor(max_workers=num_workers) as executor:
    futures = [executor.submit(wrapped_process_file, item) for item in all_files]
    for future in cf.as_completed(futures):
        segments, logs = future.result()
        X_info.extend(segments)
        logs_all.extend(logs)

print("\n전처리 완료")

if len(X_info) == 0:
    print("전처리 결과가 비어있습니다.")
    pd.DataFrame(logs_all).to_csv(os.path.join(args.output_dir, "segment_logs.csv"), index=False)
    raise ValueError("전처리 결과 없음")

# segment 파일 저장 완료 후 numpy 변환 추가 ========================

if len(X_info) == 0:
    print("전처리 결과가 비어있습니다.")
    pd.DataFrame(logs_all).to_csv(os.path.join(args.output_dir, "segment_logs.csv"), index=False)
    raise ValueError("전처리 결과 없음")

# file_names.npy, labels.npy 저장 (robust_v7 원본 저장)
file_names, labels = zip(*X_info)
np.save(os.path.join(args.output_dir, "file_names.npy"), np.array(file_names))
np.save(os.path.join(args.output_dir, "labels.npy"), np.array(labels))
pd.DataFrame(logs_all).to_csv(os.path.join(args.output_dir, "segment_logs.csv"), index=False)

# 추가: 전체 통짜 numpy 변환
X_list = []
for file in file_names:
    segment_path = os.path.join(segment_dir, file)
    features = np.load(segment_path)
    X_list.append(features)

X_array = np.stack(X_list, axis=0)
y_array = np.array(labels)

# 기존 방식과 동일하게 전체 numpy 저장
np.save(os.path.join(args.output_dir, "X_cnn.npy"), X_array)
np.save(os.path.join(args.output_dir, "y_cnn.npy"), y_array)

# summary.json 기록 유지
summary = {
    'datetime': datetime.now().isoformat(),
    'seed': SEED,
    'code_version': CODE_VERSION,
    'segment_duration': args.segment_duration,
    'sample_rate': args.target_sr,
    'hop_length': args.hop_length,
    'n_mfcc': args.n_mfcc,
    'n_fft': args.n_fft,
    'max_len': max_len,
    'label_map': label_map,
    'augmentation_factor': AUGMENTATION_FACTOR,
    'total_segments': len(X_info),
    'failed_segments': len([l for l in logs_all if 'error' in l])
}
with open(os.path.join(args.output_dir, "summary.json"), 'w') as f:
    json.dump(summary, f, indent=4)

print("전처리 및 numpy 저장까지 전체 완료!")