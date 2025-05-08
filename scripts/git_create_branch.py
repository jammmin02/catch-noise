import subprocess

def get_changed_files():
    result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    lines = result.stdout.strip().split('\n')
    return [line[3:] for line in lines if line]

def interactive_add(files):
    print("\n 변경된 파일 목록:")
    for i, file in enumerate(files):
        print(f"[{i}] {file}")
    selected = input("\n추가할 파일 번호 입력 (쉼표로 구분, 전체는 a): ").strip()
    if selected.lower() == 'a':
        for file in files:
            subprocess.run(["git", "add", file])
    else:
        indices = [int(i) for i in selected.split(',') if i.isdigit()]
        for i in indices:
            subprocess.run(["git", "add", files[i]])

def main():
    branch_type = input("브랜치 유형 입력 (예: feature, fix, docs, chore): ").strip()
    branch_name = input("작업 이름 입력 (예: update-mlflow-script): ").strip()
    full_branch = f"{branch_type}/{branch_name}"

    try:
        subprocess.run(["git", "checkout", "-b", full_branch], check=True)
        print(f"\n 브랜치 생성: {full_branch}")
    except subprocess.CalledProcessError:
        print(f"\n 브랜치 생성 실패: {full_branch} (이미 존재하거나 git 오류)")

    prefix = {
        "feature": "feat",
        "fix": "fix",
        "docs": "docs",
        "chore": "chore"
    }.get(branch_type.lower(), branch_type.lower())

    print("\n 커밋 메시지 템플릿")
    print(f"{prefix}: {branch_name.replace('-', ' ')} (by jungmin)")

    changed_files = get_changed_files()
    if not changed_files:
        print("\n 변경된 파일이 없습니다.")
    else:
        interactive_add(changed_files)
        commit_msg = input("\n 커밋 메시지 입력: ")
        subprocess.run(["git", "commit", "-m", commit_msg])

if __name__ == "__main__":
    main()
