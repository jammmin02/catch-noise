import subprocess  # 쉘 명령어(git 등)를 파이썬에서 실행하기 위한 모듈

# ✅ 로컬 브랜치 목록을 가져오는 함수
def get_local_branches():
    result = subprocess.run(["git", "branch"], capture_output=True, text=True)  # git branch 실행
    # 현재 브랜치에 '*' 표시가 있으므로 제거
    branches = [line.strip().replace("* ", "") for line in result.stdout.strip().split('\\n')]
    return branches  # 브랜치 이름 리스트 반환

# ✅ 특정 브랜치로 이동
def checkout_branch(branch):
    subprocess.run(["git", "checkout", branch])  # git checkout <branch>

# ✅ 변경된 파일 목록 가져오기 (스테이징되지 않은 파일)
def get_changed_files():
    result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    lines = result.stdout.strip().split('\\n')
    # 상태 코드를 제외한 파일 경로만 추출
    return [line[3:] for line in lines if line]

# ✅ 변경된 파일 중에서 사용자가 선택한 파일만 git add
def interactive_add(files):
    print("\\n 변경된 파일 목록:")
    for i, file in enumerate(files):
        print(f"[{i}] {file}")  # 각 파일 번호와 함께 출력

    # 파일 번호 입력 (쉼표 구분) 또는 'a'로 전체 선택
    selected = input("\\n추가할 파일 번호 입력 (쉼표로 구분, 전체는 a): ").strip()
    if selected.lower() == 'a':
        for file in files:
            subprocess.run(["git", "add", file])
    else:
        # 입력한 번호만 git add 실행
        indices = [int(i) for i in selected.split(',') if i.isdigit()]
        for i in indices:
            subprocess.run(["git", "add", files[i]])

# ✅ 메인 실행 함수
def main():
    # 🔸 브랜치 선택
    branches = get_local_branches()
    print("\\n 작업할 브랜치를 선택하세요:")
    for i, b in enumerate(branches):
        print(f"[{i}] {b}")  # 번호와 함께 브랜치 출력

    selected = input("\\n선택 (번호): ").strip()
    if not selected.isdigit() or int(selected) not in range(len(branches)):
        print(" 잘못된 입력입니다.")
        return

    branch = branches[int(selected)]  # 선택한 브랜치 이름
    checkout_branch(branch)  # 선택한 브랜치로 이동

    # 🔸 변경된 파일 확인
    changed_files = get_changed_files()
    if not changed_files:
        print(" 변경된 파일이 없습니다.")
        return

    # 🔸 파일 선택적 git add
    interactive_add(changed_files)

    # 🔸 커밋 메시지 입력 및 커밋
    commit_msg = input("\\n 커밋 메시지 입력: ")
    subprocess.run(["git", "commit", "-m", commit_msg])

    # 🔸 푸시 여부 묻기
    do_push = input(f" 브랜치 '{branch}'를 원격 저장소에 푸시할까요? (y/n): ").strip().lower()
    if do_push == 'y':
        subprocess.run(["git", "push", "origin", branch])
        print(" 푸시 완료!") 
    else:
        print(" 푸시는 생략했습니다.")

# ✅ 스크립트를 직접 실행한 경우 main() 호출
if __name__ == "__main__":
    main()