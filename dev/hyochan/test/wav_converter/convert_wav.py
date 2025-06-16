from convert_to_wav import convert_to_wav
import os

input_dir = "hyochan/test/data"
output_dir = "hyochan/test/covert"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.lower().endswith((".mp3", ".m4a", ".flac")):
        src_path = os.path.join(input_dir, filename)
        base_name = os.path.splitext(filename)[0]
        dst_path = os.path.join(output_dir, base_name + ".wav")

        convert_to_wav(src_path, dst_path)
