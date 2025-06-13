# import torchaudio
# import os

# base_dir = "data"
# for label in ["noisy", "non_noisy"]:
#     folder = os.path.join(base_dir, label)
#     for file in os.listdir(folder):
#         path = os.path.join(folder, file)
#         info = torchaudio.info(path)
#         dur = info.num_frames / info.sample_rate
#         print(f"{file}: {dur:.2f} sec")
# import torchaudio
# file_path = "/app/data/noisy/noise_01.wav"
# info = torchaudio.info(file_path)
# print(info)
# 여기를 아주 간단히 따로 만들어서 test.py로 실행
# import torchaudio
# import os

# base_dir = os.path.abspath("data")

# for label in os.listdir(base_dir):
#     folder_path = os.path.join(base_dir, label)
#     for file in os.listdir(folder_path):
#         if not file.endswith('.wav'):
#             continue
#         file_path = os.path.join(folder_path, file)
#         file_path = os.path.abspath(file_path)
#         try:
#             info = torchaudio.info(file_path)
#             print(f"{file_path} → num_frames: {info.num_frames}, sample_rate: {info.sample_rate}")
#         except Exception as e:
#             print(f"❌ Failed: {file_path} → {str(e)}")
import os
import librosa

base_dir = "data"

for root, dirs, files in os.walk(base_dir):
    for file in files:
        path = os.path.join(root, file)
        try:
            y, sr = librosa.load(path, sr=22050)
            print(f"✅ Success: {path}")
        except Exception as e:
            print(f"❌ Error at {path}: {e}")
