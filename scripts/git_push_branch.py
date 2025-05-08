import subprocess  # 쉘 명령어 (git 등)를 파이썬에서 실행하기 위해 사용

# ✅ 현재 로컬에 존재하는 git 브랜치 목록 가져오는 함수
def get_local_branches():
    result = subprocess.run(["git", "branch"], capture_output=True, text=True)  # git branch 실행 결과 받아오기
    # '* '로 표시된 현재 브랜치에서 별표 제거 후 정리
    branches = [line.strip().replace("* ", "") for line in result.stdout.strip().split('\\n')]
    return branches  # 브랜치 이름 리스트 반환

# ✅ 메인 실행 함수
def main():
    branches = get_local_branches()  # 브랜치 목록 가져오기

    # 사용자에게 브랜치 목록 출력
    print("\\n📦 푸시할 브랜치를 선택하세요:")
    for i, b in enumerate(branches):
        print(f"[{i}] {b}")  # 예: [0] dev/jungmin

    # 사용자로부터 선택 번호 입력 받기
    selected = input("\\n선택 (번호): ").strip()

    # 유효성 검사: 숫자이고 인덱스 범위 내인지 확인
    if not selected.isdigit() or int(selected) not in range(len(branches)):
        print("❌ 잘못된 입력입니다.")
        return  # 잘못된 경우 종료

    branch = branches[int(selected)]  # 선택한 브랜치 이름

    # ✅ 선택한 브랜치로 체크아웃
    subprocess.run(["git", "checkout", branch])

    # ✅ 원격 저장소로 브랜치 푸시
    subprocess.run(["git", "push", "origin", branch])

    # ✅ 완료 메시지 출력
    print(f"✅ 브랜치 '{branch}'가 원격 저장소에 푸시되었습니다.")

# ✅ 스크립트 직접 실행 시 main() 호출
if __name__ == "__main__":
    main()