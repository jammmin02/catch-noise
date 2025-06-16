import sounddevice as sd
import numpy as np
import librosa
import time
import csv
import threading
import datetime
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

# ==== 설정 ====
sr = 22050
duration = 1.0
n_mfcc = 13
hop_length = 512
frame_per_second = sr / hop_length
max_len = int(frame_per_second * duration)
input_shape = (1, 1, max_len, n_mfcc)

trt_engine_path = "dev/jungmin/jetson/assets/best_model.trt"
output_csv = f"dev/jungmin/2class_noisy_vs_nonnoisy/pyTorch_v2/outputs/cnn_lstm/realtime_eval_{datetime.datetime.now().strftime('%H%M%S')}.csv"

# ==== 상태 변수 ====
current_label = None  # 유지되는 라벨
stop_flag = False
history = []

# ==== 키보드 입력 스레드 ====
def keyboard_listener():
    global current_label, stop_flag
    print("[INFO] Press 'n' (non_noisy), 'm' (noisy), or 'q' to quit")
    while not stop_flag:
        key = input().strip().lower()
        if key == 'n':
            current_label = 0
            print("[LABEL] Set to: non_noisy (0)")
        elif key == 'm':
            current_label = 1
            print("[LABEL] Set to: noisy (1)")
        elif key == 'q':
            stop_flag = True
            print("[INFO] Stopping realtime evaluation...")
            break

# ==== 특징 추출 ====
def extract_features(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(y=audio, hop_length=hop_length)
    features = np.concatenate([mfcc, zcr], axis=0).T
    padded = np.zeros((max_len, n_mfcc))
    length = min(max_len, features.shape[0])
    padded[:length] = features[:length]
    return padded[np.newaxis, np.newaxis, :, :]

# ==== TensorRT 로딩 ====
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
with open(trt_engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

d_input = cuda.mem_alloc(trt.volume(input_shape) * np.float32().nbytes)
d_output = cuda.mem_alloc(4)  # float32 하나

host_input = np.zeros(input_shape, dtype=np.float32)
host_output = np.zeros((1,), dtype=np.float32)
bindings = [int(d_input), int(d_output)]

# ==== 시각화 준비 ====
plt.ion()
fig, ax = plt.subplots()
bar = ax.bar(["non_noisy", "noisy"], [0, 0])
ax.set_ylim([0, 1])
title = ax.set_title("Realtime Noise Prediction")
fig.canvas.draw()
fig.show()

# ==== 키보드 입력 스레드 시작 ====
threading.Thread(target=keyboard_listener, daemon=True).start()

# ==== 추론 루프 ====
try:
    while not stop_flag:
        audio = sd.rec(int(sr * duration), samplerate=sr, channels=1, dtype='float32')
        sd.wait()
        audio = audio.flatten()

        features = extract_features(audio).astype(np.float32)
        host_input[...] = features
        cuda.memcpy_htod(d_input, host_input)
        context.execute_v2(bindings)
        cuda.memcpy_dtoh(host_output, d_output)

        prob = float(host_output[0])
        pred = int(prob > 0.5)

        # 시각화
        bar[0].set_height(1 - prob)
        bar[1].set_height(prob)
        title.set_text(f"Prediction: {pred} | Label: {current_label}")
        fig.canvas.draw()
        fig.canvas.flush_events()

        # 평가 결과 출력
        if current_label is not None:
            correct = (pred == current_label)
            print(f"[EVAL] Label: {current_label}, Pred: {pred}, Prob: {prob:.3f} => {'OK' if correct else 'Wrong'}")

            history.append({
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                "label": current_label,
                "pred": pred,
                "prob": round(prob, 4),
                "correct": int(correct)
            })

        time.sleep(0.1)

except KeyboardInterrupt:
    stop_flag = True
    print("[INFO] Interrupted manually. Saving results...")

# ==== 저장 ====
if history:
    keys = history[0].keys()
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)
    print(f"[DONE] Saved evaluation results to {output_csv}")
else:
    print("[DONE] No data to save!")
