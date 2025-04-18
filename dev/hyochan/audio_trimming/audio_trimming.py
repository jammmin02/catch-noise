import os
import subprocess
import zipfile

# 설정
input_dir = "../model_make_test"  # 오디오 파일들이 들어있는 폴더
output_dir = "trimming_file"      # 자른 파일 저장 폴더
zip_output_path = "output_segments.zip"  # 최종 zip 경로
segment_duration = 2              # 자를 시간(초)

# 저장 폴더 만들기
os.makedirs(output_dir, exist_ok=True)

# 1. 오디오 파일 자르기
for file in os.listdir(input_dir):
    input_path = os.path.join(input_dir, file)
    if not os.path.isfile(input_path):
        continue
    if not file.lower().endswith(('.mp3', '.wav', '.flac', '.ogg', '.m4a')):
        continue

    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", input_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    try:
        duration = float(result.stdout.decode().strip())
    except:
        print(f"❌ 길이 추출 실패: {file}")
        continue

    num_segments = int(duration // segment_duration)

    for i in range(num_segments):
        start = i * segment_duration
        base_name = os.path.splitext(file)[0]
        output_filename = f"{base_name}_seg{i+1}.wav"
        output_path = os.path.join(output_dir, output_filename)

        subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-ss", str(start),
            "-t", str(segment_duration),
            "-ac", "1",
            "-ar", "16000",
            output_path
        ])
        print(f"✅ 저장됨: {output_filename}")

# 2. zip으로 묶기
with zipfile.ZipFile(zip_output_path, 'w') as zipf:
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, output_dir)  # zip 내 상대 경로
            zipf.write(file_path, arcname)

print(f"\n📦 압축 완료: {zip_output_path}")
