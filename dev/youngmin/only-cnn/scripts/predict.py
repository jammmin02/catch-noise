# -*- coding: utf-8 -*-
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import threading
import time
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
from python_speech_features import mfcc

# ==== 설정 ====
mic_sr = 22050
segment_duration = 2.0
segment_len = int(mic_sr * segment_duration)
n_mfcc = 13
hop_length = 512
max_len = 86
class_names = ['non_noisy', 'noisy']
engine_path = "only-cnn.engine"

# ==== TensorRT 엔진 로드 ====
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

# 입출력 버퍼 준비
input_shape = (1, 1, 86, 13)
output_shape = (1, 1)
d_input = cuda.mem_alloc(int(np.prod(input_shape) * np.float32().nbytes))
d_output = cuda.mem_alloc(int(np.prod(output_shape) * np.float32().nbytes))
bindings = [int(d_input), int(d_output)]
stream = cuda.Stream()

# ==== 실시간 입력용 변수 ====
rolling_audio = np.zeros(segment_len, dtype=np.float32)
latest_pred = np.array([0.5, 0.5])
latest_label = "non_noisy"
latest_confidence = 0.5

# ==== 실시간 시각화 초기화 ====
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

bar_plot = ax1.bar(class_names, latest_pred, color=["skyblue", "salmon"])
ax1.set_ylim(0, 1)
ax1.set_title("Prediction results")
ax1.set_ylabel("Probability")

plot_len = int(mic_sr * 0.5)
line_wave, = ax2.plot(np.zeros(plot_len))
ax2.set_ylim(-1, 1)
ax2.set_title("Real-time microphone input")
ax2.set_xlabel("Sample")
ax2.set_ylabel("Amplitude")
plt.tight_layout()
plt.show(block=False)

# ==== 특징 추출 ====
def extract_mfcc_features(audio):
    feat = mfcc(audio, samplerate=mic_sr, numcep=n_mfcc)
    if feat.shape[0] < max_len:
        feat = np.pad(feat, ((0, max_len - feat.shape[0]), (0, 0)), mode="constant")
    else:
        feat = feat[:max_len]
    return feat[np.newaxis, np.newaxis, :, :].astype(np.float32)

# ==== 마이크 콜백 ====
def audio_callback(indata, frames, time_info, status):
    global rolling_audio
    mono = indata[:, 0]
    rolling_audio = np.roll(rolling_audio, -len(mono))
    rolling_audio[-len(mono):] = mono

# ==== 예측 함수 ====
def predict_with_trt(input_array):
    input_array = input_array.astype(np.float32).ravel()
    output_array = np.empty((1,), dtype=np.float32)

    cuda.memcpy_htod_async(d_input, input_array, stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(output_array, d_output, stream)
    stream.synchronize()

    return output_array

# ==== 예측 쓰레드 ====
def prediction_loop():

    global latest_pred, latest_label, latest_confidence
    ctx = pycuda.autoinit.context
    ctx.push()

    try:
        while True:
            time.sleep(1.0)
            x = extract_mfcc_features(rolling_audio)
            pred = predict_with_trt(x)
            prob = float(pred[0])
            latest_pred = np.array([1 - prob, prob])
            label_idx = int(prob > 0.7)
            latest_label = class_names[label_idx]
            latest_confidence = latest_pred[label_idx]
            print(f"\nPrediction: [{latest_label}] (Confidence: {latest_confidence:.2f})")
    finally:
        ctx.pop()

# ==== 스트리밍 및 쓰레드 시작 ====
print("🎧 실시간 예측 시작... Ctrl+C로 종료 가능")
audio_stream = sd.InputStream(
    samplerate=mic_sr,
    channels=1,
    dtype='float32',
    blocksize=int(mic_sr * 0.05),
    callback=audio_callback
)
audio_stream.start()
threading.Thread(target=prediction_loop, daemon=True).start()

# ==== 시각화 루프 ====
try:
    while True:
        line_wave.set_ydata(rolling_audio[-plot_len:])
        for i, bar in enumerate(bar_plot):
            bar.set_height(latest_pred[i])
        ax1.set_title(f"Prediction: [{latest_label}] (Confidence: {latest_confidence:.2f})")
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.05)

except KeyboardInterrupt:
    print("\n⛔ 종료됨")
    audio_stream.stop()
    plt.ioff()
    plt.close()
