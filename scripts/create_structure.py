import os

# 생성할 디렉토리 구조 정의
dirs = [
    ".github",
    "data",
    "dev/azuki",
    "dev/hyochan",
    "dev/hyunwoo",
    "dev/jungmin",
    "dev/youngmin",
    "docker",
    "models",
    "outputs",
    "scripts",
    "src/dataset",
    "src/model",
    "src/predict",
    "src/train",
    "test_audio_batch"
]

# 디렉토리 생성 함수
def create_dirs():
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"✅ Created: {d}")

# 기본 파일 생성
def create_files():
    open(".gitignore", "w").write("__pycache__/\n*.pyc\n.env\n*.pth\noutputs/\nmodels/\n")
    open("README.md", "w").write("# team-noise-ai-project\n\nNoise classification project for classroom AI system.")
    os.makedirs(".github", exist_ok=True)
    open(".github/PULL_REQUEST_TEMPLATE.md", "w").write("""## \u2705 \ubcc0\uacbd \uc0ac\ud56d \uc694\uc57d\n- \uc774 PR\uc740 \ubb34\uc5c7\uc744 \ud558\ub098\uc694?\n\n## \ud310\ub2e8 \ucee8\uc11c\ud2b8\n- [ ] \ucf54\ub4dc\uac00 \uc798 \ub3fc\uc788\ub098\uc694?\n- [ ] \uad00\ub828\ub41c README \ubb38\uc11c\ub97c \uc218\uc815\ud588\ub098\uc694?\n""")

if __name__ == "__main__":
    print("\n📁 프로젝트 디렉토리 생성 중...\n")
    create_dirs()
    create_files()
    print("\n🎉 모든 디렉토리와 기본 파일 생성 완료!")
