import os

# 생성할 폴더 구조 정의
FOLDER_STRUCTURE = [
    ".github",
    "dev/azuki",
    "dev/hyomin",
    "dev/hyunwoo",
    "dev/youngmin",
    "dev/hyochan",
    "dev/yeongmin",
    "data",
    "docker",
    "models",
    "outputs",
    "scripts",
    "src/dataset",
    "src/model",
    "src/predict",
    "src/train",
    "test"
]

def create_folders(base_path="."):
    for folder in FOLDER_STRUCTURE:
        folder_path = os.path.join(base_path, folder)
        os.makedirs(folder_path, exist_ok=True)

        # .gitkeep 파일 추가
        gitkeep_path = os.path.join(folder_path, ".gitkeep")
        with open(gitkeep_path, "w") as f:
            pass  # 빈 파일

    print("✅ 폴더 구조와 .gitkeep 파일이 생성되었습니다!")

if __name__ == "__main__":
    create_folders()
