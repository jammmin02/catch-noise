import os
import subprocess

# 설정
input_path = "../model_make_test"  # 원본 파일 경로 (같은 폴더에 있으면 파일명만 써도 됨)
output_dir = "trimming_file"  # 저장 폴더 이름
segment_duration = 2  # 자를 시간(초)

# 저장 폴더 만들기
os.makedirs(output_dir, exist_ok=True)

# 오디오 길이 가져오기 (초 단위)
result = subprocess.run(
    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
     "-of", "default=noprint_wrappers=1:nokey=1", input_path],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT
)
duration = float(result.stdout.decode().strip())
num_segments = int(duration // segment_duration)

# ffmpeg로 자르기 + mono, 16kHz 변환
for i in range(num_segments):
    start = i * segment_duration
    output_filename = f"{os.path.splitext(os.path.basename(input_path))[0]}_seg{i+1}.wav"
    output_path = os.path.join(output_dir, output_filename)

    subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-ss", str(start),
        "-t", str(segment_duration),
        "-ac", "1",  # mono
        "-ar", "16000",  # 16kHz
        output_path
    ])
