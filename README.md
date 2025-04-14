# 🔊 TEAM-NOISE-AI-PROJECT

**Noise classification project for classroom AI system**  
AI를 활용해 교실 소음을 실시간으로 분석하고, 학습을 방해하는 소리와 방해하지 않는 소리를 구분하여 시각적으로 표시하는 프로젝트입니다.  
학생 스스로 소음 환경을 인지하고 조절할 수 있도록 돕는 **자율 학습 환경 구축**을 목표로 합니다.

<br>

## 📂 REPOSITORY 구 조 도

<img src="https://github.com/HyoChan1117/HyoChan1117/raw/master/team_project-structure.drawio.png" alt="Project Structure" width="80%">

<br>

## 📁 디렉토리 설명

<table>
  <tr>
    <th>디렉토리</th>
    <th>설명</th>
  </tr>
  <tr>
    <td><code>src/</code></td>
    <td>학습, 예측, 모델 정의 등 공통 코드</td>
  </tr>
  <tr>
    <td><code>dev/</code></td>
    <td>팀원별 실험 공간 (브랜치 기반)</td>
  </tr>
  <tr>
    <td><code>models/</code></td>
    <td>학습된 모델 저장 (.pth 등)</td>
  </tr>
  <tr>
    <td><code>outputs/</code></td>
    <td>시각화, 로그, 평가 결과 저장</td>
  </tr>
  <tr>
    <td><code>data/</code></td>
    <td>공통 오디오 데이터셋</td>
  </tr>
  <tr>
    <td><code>docker/</code></td>
    <td>Docker 실행 환경 파일</td>
  </tr>
  <tr>
    <td><code>scripts/</code></td>
    <td>유틸 스크립트, 정리 도구</td>
  </tr>
  <tr>
    <td><code>test/</code></td>
    <td>샘플 테스트 오디오</td>
  </tr>
  <tr>
    <td><code>.gitignore</code></td>
    <td>Git 추적 제외 항목 설정</td>
  </tr>
</table>

<br>

## ⚙️ 사용 기술 스택

| 분류 | 기술 |
|------|------|
| 언어 | Python |
| 프레임워크 | PyTorch |
| 도구 | Docker, GitHub Actions, Shell Script |
| 라이브러리 | Librosa, OpenCV, Matplotlib |
| 특징 추출 | MFCC, ZCR |
| 모델 구조 | CNN2D + LSTM (Sequential Classification) |

<br>

## 🎯 프로젝트 목표

| 목적 | 설명 |
|------|------|
| 🔉 소음 분류 | 교실 내 소리를 분석해 `조용한 소리` / `시끄러운 소리`로 실시간 분류 |
| 🎛️ 모드 전환 | 3가지 모드 제공: `도서관`, `회의`, `쉬는 시간`에 따라 허용 기준 다름 |
| 🌐 웹 시각화 | 분석 결과를 웹 UI에 실시간 표시 (색상 + dB 값 등 시각 요소) |
| 🧠 사용자 피드백 기반 개선 | 사용자 피드백을 수집하여 AI 모델을 지속적으로 개선 |
| 🔁 자동 비측정 기능 | 수업 시간에는 자동으로 측정을 중단 (시간표 기반) |

<br>

## 🧑‍💻 팀 협업 규칙

| 항목 | 규칙 |
|------|------|
| 브랜치 전략 | `main`은 보호됨 → PR 통해 병합 (CODEOWNERS 승인 필요) |
| 작업 공간 | 팀원별 `dev/이름/` 디렉토리에서 실험 후 통합 |
| 코드 병합 | 리뷰 및 테스트 완료 후 `src/`에 병합 |
| 모델 공유 | 성능 검증된 모델만 `models/`에 저장 |
| 커밋 메시지 | 팀 템플릿에 맞춰 작성 (예: `feat:`, `fix:`, `docs:` 등) |

<br>

## 🧪 프로젝트 주요 특징

| 항목 | 내용 |
|------|------|
| 🎤 마이크 | 원거리 수음 가능한 무지향성 마이크 사용 |
| 🧱 하드웨어 | Jetson Nano 기반 실시간 오디오 처리 |
| 🔀 데이터 | 라벨링 기준은 수집자 주관 + 피드백 반영 |
| 📊 평가 방식 | 실험용 오디오를 활용한 정확도/혼동 행렬 기반 평가 |
| 🐳 실행 환경 | Docker로 모든 실행 환경 일관성 확보 |

<br>

📌 프로젝트 상세 문서는 곧 `docs/` 디렉토리에 추가될 예정입니다.  
📬 제안이나 문의는 **Issue** 또는 **Pull Request**로 환영합니다!
