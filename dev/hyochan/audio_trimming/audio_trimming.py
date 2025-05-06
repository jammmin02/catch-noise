import os
import subprocess
import zipfile

# 설정
input_dir = "hyochan/model_make_test/data"  # 모든 클래스 폴더가 있는 상위 경로
output_dir = "hyochan/audio_trimming/trimming_file"              # 잘린 오디오 저장 경로
zip_output_path = "output_segments.zip"   # 압축 파일명
segment_duration = 2                      # 자를 시간 (초)

# 출력 폴더 생성
os.makedirs(output_dir, exist_ok=True)

# 모든 하위 클래스 폴더 순회
for class_name in os.listdir(input_dir):
    class_path = os.path.join(input_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    for file in os.listdir(class_path):
        input_path = os.path.join(class_path, file)
        if not os.path.isfile(input_path):
            continue
        if not file.lower().endswith(('.mp3', '.wav', '.flac', '.ogg', '.m4a')):
            continue

        # 오디오 길이 측정
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
            segment_name = f"{base_name}_seg{i+1}.wav"
            class_output_dir = os.path.join(output_dir, class_name)
            os.makedirs(class_output_dir, exist_ok=True)
            output_path = os.path.join(class_output_dir, segment_name)

            subprocess.run([
                "ffmpeg", "-y", "-i", input_path,
                "-ss", str(start),
                "-t", str(segment_duration),
                "-ac", "1",
                "-ar", "16000",
                output_path
            ])
            print(f"✅ 저장됨: {class_name}/{segment_name}")

# zip 압축 (폴더 구조 유지)
with zipfile.ZipFile(zip_output_path, 'w') as zipf:
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, output_dir)  # zip 내부 구조
            zipf.write(file_path, arcname)

print(f"\n📦 압축 완료: {zip_output_path}")
