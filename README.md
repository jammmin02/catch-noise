# 🔊 CATCH-NOISE-AI-PROJECT

**Noise classification project for classroom AI system**  
AI를 활용해 교실 소음을 실시간으로 분석하고, 학습을 방해하는 소리와 방해하지 않는 소리를 구분하여 시각적으로 표시하는 프로젝트입니다.  
학생 스스로 소음 환경을 인지하고 조절할 수 있도록 돕는 **자율 학습 환경 구축**을 목표로 합니다.

<br>

## 👥 팀원 소개

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/jammmin02.png" width="100px;" alt="박정민"/><br />
      <sub><b>박정민</b></sub><br />
      팀장<br />
      <a href="https://github.com/jammmin02" target="_blank">@jammmin02</a>
    </td>
    <td align="center">
      <img src="https://github.com/HyoChan1117.png" width="100px;" alt="김효찬"/><br />
      <sub><b>김효찬</b></sub><br />
      팀원<br />
      <a href="https://github.com/HyoChan1117" target="_blank">@HyoChan1117</a>
    </td>
    <td align="center">
      <img src="https://github.com/youngmin109.png" width="100px;" alt="배영민"/><br />
      <sub><b>배영민</b></sub><br />
      팀원<br />
      <a href="https://github.com/youngmin109" target="_blank">@youngmin109</a>
    </td>
    <td align="center">
      <img src="https://github.com/gould7789.png" width="100px;" alt="이현우"/><br />
      <sub><b>이현우</b></sub><br />
      팀원<br />
      <a href="https://github.com/gould7789" target="_blank">@gould7789</a>
    </td>
    <td align="center">
      <img src="https://github.com/Azuking69.png" width="100px;" alt="아즈키"/><br />
      <sub><b>아즈키</b></sub><br />
      팀원<br />
      <a href="https://github.com/Azuking69" target="_blank">@Azuking69</a>
    </td>
  </tr>
</table>


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

## 🧩 팀 공통 개발 환경 & 라이브러리 버전 / Team-wide Dev Environment & Library Versions

| 항목 | 내용 |
|------|------|
| **기반 이미지** | `pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel`<br>`nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04` |
| **CUDA 버전** | `10.2`, `11.7.1` (사용 환경에 따라 선택) |
| **Python 버전** | `Python 3.6` (기본 설치 후 `/usr/bin/python` 링크 연결) |
| **주요 시스템 패키지** | `ffmpeg`, `libsndfile1`, `portaudio19-dev`, `libsm6`, `libxrender-dev`, `cmake`, `git` 등 |
| **추가 키 설정** | NVIDIA CUDA GPG 키 수동 등록 |
| **기본 실행 명령어** | `CMD ["bash"]` |

### 🔧 PyTorch 생태계 라이브러리

| 패키지 | 버전 | 주석 |
|--------|-------|------|
| `torch` | `1.13.1+cu117` | CUDA 11.7 대응 버전 |
| `torchvision` | `0.14.1+cu117` | |
| `torchaudio` | `0.13.1` | |
| `torch` | `1.9.0` | (추가로 명시됨) |
| `torchvision` | `0.10.0` | (추가로 명시됨) |

### 📦 주요 requirements.txt 패키지 요약

| 범주 | 패키지 | 버전 |
|------|--------|------|
| **기초 라이브러리** | `numpy`, `pandas`, `scikit-learn` | `1.22.4`, `1.3.5`, `1.0.2` |
| **시각화** | `matplotlib`, `seaborn`, `pyqt5`, `pyqtgraph` | `3.5.2`, `0.11.2`, 기타 |
| **오디오 처리** | `librosa`, `soundfile`, `pyaudio`, `scipy`, `sounddevice` | `0.9.2`, `0.10.3.post1`, `0.4.6`, 기타 |
| **실험 관리/최적화** | `mlflow`, `optuna`, `tqdm` | `1.30.0`, `3.0.3`, `4.64.1` |
| **기타 유틸** | `imageio`, `opencv-python`, `Cython<3` | `2.19.3`, 등 |

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
