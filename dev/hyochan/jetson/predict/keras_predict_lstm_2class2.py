# -*- coding: utf-8 -*-
import sounddevice as sd
import librosa
import numpy as np
import matplotlib.pyplot as plt
import threading
import time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # 메인 스레드에서 context 자동 생성

# === 설정 ===
mic_sr = 44100
model_sr = 22050
segment_duration = 2.0
n_mfcc = 13
hop_length = 512
max_len = 86
class_names = ['non_noisy', 'noisy']
engine_path = "hyochan/jetson_predict/cnn_lstm_model.trt"

# === TensorRT 엔진 로드 ===
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

input_shape = (1, 86, 14, 1)
input_nbytes = np.prod(input_shape) * np.float32().nbytes
output_nbytes = 1 * np.float32().nbytes  # 시그모이드 출력 1개

# === 마이크 장치 자동 탐색 ===
def find_input_device(name_keyword="usb"):
    for idx, dev in enumerate(sd.query_devices()):
        if name_keyword.lower() in dev["name"].lower() and dev["max_input_channels"] > 0:
            print(f"Selected microphone: {dev['name']} (index {idx})")
            return idx, dev["max_input_channels"]
    raise RuntimeError("USB microphone not found.")

device_index, device_channels = find_input_device("usb")

# === 실시간 변수 ===
rolling_audio = np.zeros(int(mic_sr * segment_duration), dtype=np.float32)
latest_pred = np.array([0.5, 0.5])
latest_label = "non_noisy"
latest_confidence = 0.5

# === 시각화 초기화 ===
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

# === 특징 추출 ===
def extract_features(y_audio):
    y_audio = librosa.resample(y_audio, orig_sr=mic_sr, target_sr=model_sr)
    mfcc = librosa.feature.mfcc(y=y_audio, sr=model_sr, n_mfcc=n_mfcc, hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(y=y_audio, hop_length=hop_length)
    features = np.vstack([mfcc, zcr])
    if features.shape[1] < max_len:
        features = np.pad(features, ((0, 0), (0, max_len - features.shape[1])), mode='constant')
    else:
        features = features[:, :max_len]
    return features.T[np.newaxis, ..., np.newaxis].astype(np.float32)

# === 마이크 콜백 ===
def audio_callback(indata, frames, time_info, status):
    global rolling_audio
    mono_input = indata[:, 0]
    rolling_audio = np.roll(rolling_audio, -len(mono_input))
    rolling_audio[-len(mono_input):] = mono_input

# === TensorRT 예측 함수 ===
def predict_with_trt(input_array):
    input_array = input_array.astype(np.float32).ravel()
    output_array = np.empty((1,), dtype=np.float32)

    context = engine.create_execution_context()
    stream = cuda.Stream()
    d_input = cuda.mem_alloc(int(input_nbytes))
    d_output = cuda.mem_alloc(int(output_nbytes))
    bindings = [int(d_input), int(d_output)]

    cuda.memcpy_htod_async(d_input, input_array, stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(output_array, d_output, stream)
    stream.synchronize()

    return output_array

# === 예측 쓰레드 ===
def predict_thread():
    global rolling_audio, latest_pred, latest_label, latest_confidence
    segment_len = int(mic_sr * segment_duration)
    ctx = pycuda.autoinit.context
    ctx.push()  # context push

    try:
        while True:
            time.sleep(1.0)
            audio_seg = rolling_audio[-segment_len:]
            x = extract_features(audio_seg)
            pred = predict_with_trt(x)
            p = float(pred[0])  # sigmoid 1개
            latest_pred = np.array([1 - p, p])
            label_idx = int(p > 0.5)
            latest_label = class_names[label_idx]
            latest_confidence = latest_pred[label_idx]
            print(f"\nPrediction: [{latest_label}] (Confidence: {latest_confidence:.2f})")
    finally:
        ctx.pop()  # context pop

# === 마이크 스트리밍 시작 ===
print("Real-time prediction started. Press Ctrl+C to stop.")
audio_stream = sd.InputStream(
    device=device_index,
    callback=audio_callback,
    channels=1,
    samplerate=mic_sr,
    blocksize=int(mic_sr * 0.05)
)
audio_stream.start()

threading.Thread(target=predict_thread, daemon=True).start()

# === 메인 루프 (시각화) ===
try:
    while True:
        line_wave.set_ydata(rolling_audio[-plot_len:])
        for i, bar in enumerate(bar_plot):
            bar.set_height(latest_pred[i])
        ax1.set_title(f"Prediction result: [{latest_label}] (Confidence: {latest_confidence:.2f})")
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nTerminated by user.")
    audio_stream.stop()
    plt.ioff()
    plt.close()
