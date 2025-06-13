import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import sounddevice as sd
import librosa
import time

# 설정
mic_sr = 44100
model_sr = 22050
segment_duration = 2.0
n_mfcc = 13
hop_length = 512
max_len = 86

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
engine_file_path = "cnn_only.trt"

# 특징 추출 함수 (librosa 기반)
def extract_features(y_audio):
    y_audio = librosa.resample(y_audio, orig_sr=mic_sr, target_sr=model_sr)
    mfcc = librosa.feature.mfcc(y=y_audio, sr=model_sr, n_mfcc=n_mfcc, hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(y=y_audio, hop_length=hop_length)
    features = np.vstack([mfcc, zcr])

    if features.shape[1] < max_len:
        features = np.pad(features, ((0, 0), (0, max_len - features.shape[1])), mode='constant')
    else:
        features = features[:, :max_len]

    features = features.T[np.newaxis, ..., np.newaxis].astype(np.float32)
    features = np.transpose(features, (0, 3, 1, 2))  # (1, 1, 86, 14)
    return features

# TensorRT 로드
with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f)
    context = engine.create_execution_context()

# I/O 버퍼 준비
input_shape = (1, 1, 86, 14)
output_shape = (1, 1)

d_input = cuda.mem_alloc(np.prod(input_shape) * np.float32().nbytes)
d_output = cuda.mem_alloc(np.prod(output_shape) * np.float32().nbytes)
bindings = [int(d_input), int(d_output)]

# 실시간 마이크 스트리밍
print("TensorRT 실시간 추론 시작")

rolling_audio = np.zeros(int(mic_sr * segment_duration), dtype=np.float32)

def callback(indata, frames, time_info, status):
    global rolling_audio
    mono_input = indata[:, 0]
    rolling_audio = np.roll(rolling_audio, -len(mono_input))
    rolling_audio[-len(mono_input):] = mono_input

stream = sd.InputStream(callback=callback, channels=1, samplerate=mic_sr, blocksize=int(mic_sr * 0.05))
stream.start()

try:
    while True:
        time.sleep(1.0)
        audio_seg = rolling_audio[-int(mic_sr * segment_duration):]
        features = extract_features(audio_seg)

        cuda.memcpy_htod(d_input, features)
        context.execute_v2(bindings)
        output = np.empty(output_shape, dtype=np.float32)
        cuda.memcpy_dtoh(output, d_output)

        p = float(output[0, 0])
        label = "noisy" if p > 0.5 else "non_noisy"
        print(f"Prediction: [{label}] (Confidence: {p:.2f})")

except KeyboardInterrupt:
    print("\n종료됨")
    stream.stop()
