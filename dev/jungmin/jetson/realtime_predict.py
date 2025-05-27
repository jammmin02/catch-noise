import sounddevice as sd
import numpy as np
import librosa
import time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")  # GUI matplotlib backend

# -------- Configuration --------
sr = 22050
duration = 1.0  # seconds
n_mfcc = 13
hop_length = 512
frame_per_second = sr / hop_length
max_len = int(frame_per_second * duration)
input_shape = (1, 1, max_len, n_mfcc)  # (B, C, H, W)

trt_engine_path = "dev/jungmin/jetson/assets/best_model.trt"

# -------- Feature extraction --------
def extract_features(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(y=audio, hop_length=hop_length)
    features = np.concatenate([mfcc, zcr], axis=0).T  # (T, C)
    padded = np.zeros((max_len, n_mfcc))
    length = min(max_len, features.shape[0])
    padded[:length] = features[:length]
    return padded[np.newaxis, np.newaxis, :, :]  # (1, 1, H, W)

# -------- Load TensorRT engine --------
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
with open(trt_engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

input_binding_idx = engine.get_binding_index("input")
output_binding_idx = engine.get_binding_index("output")
output_shape = (1, 1)

d_input = cuda.mem_alloc(trt.volume(input_shape) * np.float32().nbytes)
d_output = cuda.mem_alloc(trt.volume(output_shape) * np.float32().nbytes)

host_input = np.zeros(input_shape, dtype=np.float32)
host_output = np.zeros(output_shape, dtype=np.float32)

bindings = [int(d_input), int(d_output)]

# -------- Realtime visualization --------
plt.ion()
fig, ax = plt.subplots()
bar = ax.bar(["non_noisy", "noisy"], [0, 0])
ax.set_ylim([0, 1])
ax.set_title("Realtime Noise Prediction (TensorRT)")
fig.canvas.draw()
fig.show()

# -------- Inference loop --------
print("Starting realtime microphone prediction... (Press Ctrl+C to stop)")

try:
    while True:
        audio = sd.rec(int(sr * duration), samplerate=sr, channels=1, dtype='float32')
        sd.wait()
        audio = audio.flatten()
        features = extract_features(audio).astype(np.float32)
        host_input[...] = features

        cuda.memcpy_htod(d_input, host_input)
        context.execute_v2(bindings)
        cuda.memcpy_dtoh(host_output, d_output)

        prob = float(host_output[0])
        bar[0].set_height(1 - prob)
        bar[1].set_height(prob)
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.1)

except KeyboardInterrupt:
    print("Realtime prediction stopped.")
