import torchaudio
import torchaudio.transforms as T
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

# 설정
SAMPLE_RATE = 16000
N_MFCC = 13
PLOT_COUNT = 3
FRAME_SIZE = 2048
HOP_SIZE = 512
N_CLASSES = {'noizy': 0, 'non_noizy': 1}
FIXED_LEN = 87  # 2초 기준 (16000 / 512 ≈ 31.25 → 2초면 약 87프레임)

# 경로
root_dir = Path(__file__).resolve().parent.parent
data_root = root_dir / "data"
output_root = root_dir / "torchAudio"
viz_dir = output_root / "viz_outputs"
npy_dir = output_root / "npy"
viz_dir.mkdir(parents=True, exist_ok=True)
npy_dir.mkdir(parents=True, exist_ok=True)

# 전처리기
mfcc = T.MFCC(sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC, melkwargs={"n_fft": 512, "hop_length": HOP_SIZE, "n_mels": 40})
resample = T.Resample(orig_freq=44100, new_freq=SAMPLE_RATE)

# ZCR 계산 + smoothing
def compute_smoothed_zcr(waveform, frame_size=FRAME_SIZE, hop_size=HOP_SIZE, smooth=5):
    waveform_np = waveform.numpy()[0]
    zcr = []
    for i in range(0, len(waveform_np) - frame_size, hop_size):
        frame = waveform_np[i:i+frame_size]
        crossings = np.diff(np.sign(frame))
        zcr_val = (crossings != 0).sum() / frame_size
        zcr.append(zcr_val)
    zcr = np.array(zcr)
    if smooth > 1:
        zcr = np.convolve(zcr, np.ones(smooth)/smooth, mode='same')
    return torch.tensor(zcr)

# MFCC를 0~1 정규화
def normalize_features(feat: torch.Tensor):
    scaler = MinMaxScaler()
    return torch.tensor(scaler.fit_transform(feat.T).T, dtype=torch.float)

# 시각화
def draw_figure(waveform, mfcc_feat, zcr_curve, save_path, title):
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)

    # 1. Waveform
    time = np.linspace(0, len(waveform[0]) / SAMPLE_RATE, len(waveform[0]))
    axs[0].plot(time, waveform[0].numpy(), color='gray')
    axs[0].set_title("Waveform")
    axs[0].set_ylabel("Amplitude")
    axs[0].set_xlabel("Time (s)")

    # 2. MFCC 히트맵
    mfcc_norm = normalize_features(mfcc_feat)
    im = axs[1].imshow(mfcc_norm.numpy(), origin='lower', aspect='auto', cmap='magma', vmin=0, vmax=1)
    axs[1].set_title("MFCC Heatmap (Normalized)")
    axs[1].set_ylabel("MFCC Coefficients")
    axs[1].set_xlabel("Frame")
    fig.colorbar(im, ax=axs[1])

    # 3. ZCR
    axs[2].plot(zcr_curve.numpy(), linewidth=2, color='purple')
    axs[2].set_title("ZCR Over Frames (Smoothed)")
    axs[2].set_xlabel("Frame")
    axs[2].set_ylabel("ZCR")
    axs[2].set_ylim(0, max(0.2, zcr_curve.max().item() + 0.05))

    fig.suptitle(title)
    plt.savefig(save_path)
    plt.close()


# 전체 실행
x_data = []
y_data = []

for label in N_CLASSES.keys():
    path = data_root / label
    files = list(path.glob("*.wav"))
    print(f" {label}: {len(files)}개 파일 처리 중...")

    for file in tqdm(files):
        waveform, sr = torchaudio.load(file)
        if sr != SAMPLE_RATE:
            waveform = resample(waveform)

        waveform = (waveform - waveform.mean()) / waveform.std()
        mfcc_feat = mfcc(waveform)[0]  # [n_mfcc, T]
        zcr_curve = compute_smoothed_zcr(waveform)

        # 길이 고정
        min_len = min(FIXED_LEN, mfcc_feat.shape[1], len(zcr_curve))
        mfcc_fixed = mfcc_feat[:, :min_len]
        zcr_fixed = zcr_curve[:min_len]

        # 저장용
        features = torch.cat([mfcc_fixed, zcr_fixed.unsqueeze(0)], dim=0)  # [14, T]
        x_data.append(features.numpy())
        y_data.append(N_CLASSES[label])

        # 시각화
        save_name = viz_dir / f"{label}_{file.stem}.png"
        draw_figure(waveform, mfcc_fixed, zcr_fixed, save_name, f"{file.stem} ({label})")


# .npy 저장
np.save(npy_dir / "x.npy", np.array(x_data))  # (N, 14, T)
np.save(npy_dir / "y.npy", np.array(y_data))  # (N,)
print(".npy 저장 완료 및 시각화 종료")
