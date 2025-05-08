import sounddevice as sd
import soundfile as sf

fs = 16000
duration = 3
device_index = 18  # Logitech StreamCam 마이크 (WDM-KS)

print("🎤 녹음 시작...")
recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, device=device_index)
sd.wait()
print("✅ 녹음 완료!")

sf.write("webcam_recording.wav", recording, fs)