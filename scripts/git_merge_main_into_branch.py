import subprocess

#  로컬 브랜치 목록 가져오기
def get_local_branches():
    result = subprocess.run(["git", "branch"], capture_output=True, text=True)
    branches = [line.strip().replace("* ", "") for line in result.stdout.strip().split('\n')]
    return branches

#  브랜치 체크아웃
def checkout(branch):
    subprocess.run(["git", "checkout", branch])

#  해당 브랜치 최신 pull
def pull(branch):
    subprocess.run(["git", "pull", "origin", branch])

#  병합 시도 (충돌 감지 포함)
def merge(from_branch):
    print(f"\n main을 병합 중: git merge {from_branch} --no-commit --no-ff")
    result = subprocess.run(
        ["git", "merge", from_branch, "--no-commit", "--no-ff"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("❌ 충돌이 발생했습니다!")
        print("🔧 수동으로 충돌을 해결하고, 직접 'git merge --continue'를 실행해주세요.")
        exit(1)  # 병합 실패 시 스크립트 종료
    else:
        print(" 병합 성공! 자동 커밋 진행 중...")
        subprocess.run(["git", "commit", "-m", f"Merge {from_branch} into current branch"])

#  병합된 브랜치 push
def push(branch):
    subprocess.run(["git", "push", "origin", branch])
    print(f" '{branch}' 브랜치 푸시 완료!")

#  메인 함수
def main():
    # 1. 병합 대상 브랜치 선택
    branches = get_local_branches()
    print("\n 병합 대상 브랜치를 선택하세요:")
    for i, b in enumerate(branches):
        print(f"[{i}] {b}")
    selected = input("\n선택 (번호): ").strip()

    if not selected.isdigit() or int(selected) not in range(len(branches)):
        print("❌ 잘못된 입력입니다.")
        return

    target_branch = branches[int(selected)]

    # 2. main 최신화
    print("\n main 브랜치 최신화 중...")
    checkout("main")
    pull("main")

    # 3. 대상 브랜치로 이동
    checkout(target_branch)

    # 4. main → 대상 브랜치 병합 (충돌 감지)
    merge("main")

    # 5. 푸시 여부 확인
    do_push = input(f"\n '{target_branch}' 브랜치를 원격에 푸시할까요? (y/n): ").strip().lower()
    if do_push == 'y':
        push(target_branch)
    else:
        print(" 푸시는 생략했습니다.")

# 실행 시작점
if __name__ == "__main__":
    main()
