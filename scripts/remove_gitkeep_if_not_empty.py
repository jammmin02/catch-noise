import os

def remove_gitkeep_if_not_empty(root_dir="."):
    removed_files = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # .git 디렉토리는 무시
        if ".git" in dirpath.split(os.sep):
            continue

        if ".gitkeep" in filenames:
            other_files = [f for f in filenames if f != ".gitkeep"]
            if other_files:  # 다른 파일이 존재하면
                file_path = os.path.join(dirpath, ".gitkeep")
                try:
                    os.remove(file_path)
                    removed_files.append(file_path)
                except Exception as e:
                    print(f"❌ Failed to delete {file_path}: {e}")

    if removed_files:
        print("✅ .gitkeep files removed from non-empty folders:")
        for f in removed_files:
            print(f"  - {f}")
    else:
        print("ℹ️ No .gitkeep files needed removal.")

if __name__ == "__main__":
    remove_gitkeep_if_not_empty()
