# 🔊 CATCH-NOISE-AI-PROJECT  
**Noise classification project for classroom AI system**  
**教室内の騒音をリアルタイムで分類するAIプロジェクト**

AI를 활용해 교실 소음을 실시간으로 분석하고, 학습을 방해하는 소리와 방해하지 않는 소리를 구분하여 시각적으로 표시하는 프로젝트입니다.  
AIを活用して教室内の騒音をリアルタイムに分析し、「学習を妨げる音」と「妨げない音」を区別して可視化するプロジェクトです。

학생 스스로 소음 환경을 인지하고 조절할 수 있도록 돕는 **자율 학습 환경 구축**을 목표로 합니다.  
学生自身が騒音環境を認識し、自律的にコントロールできる**自己主導型学習環境**の構築を目指しています。

---

## 👥 팀원 소개 | メンバー紹介

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="https://github.com/jammmin02.png" width="100px;" alt="박정민"/><br />
        <sub><b>박정민</b></sub><br />팀장 / リーダー<br />
        <a href="https://github.com/jammmin02" target="_blank">@jammmin02</a>
      </td>
      <td align="center">
        <img src="https://github.com/HyoChan1117.png" width="100px;" alt="김효찬"/><br />
        <sub><b>김효찬</b></sub><br />팀원 / メンバー<br />
        <a href="https://github.com/HyoChan1117" target="_blank">@HyoChan1117</a>
      </td>
      <td align="center">
        <img src="https://github.com/youngmin109.png" width="100px;" alt="배영민"/><br />
        <sub><b>배영민</b></sub><br />팀원 / メンバー<br />
        <a href="https://github.com/youngmin109" target="_blank">@youngmin109</a>
      </td>
      <td align="center">
        <img src="https://github.com/gould7789.png" width="100px;" alt="이현우"/><br />
        <sub><b>이현우</b></sub><br />팀원 / メンバー<br />
        <a href="https://github.com/gould7789" target="_blank">@gould7789</a>
      </td>
      <td align="center">
        <img src="https://github.com/Azuking69.png" width="100px;" alt="아즈키"/><br />
        <sub><b>아즈키</b></sub><br />팀원 / メンバー<br />
        <a href="https://github.com/Azuking69" target="_blank">@Azuking69</a>
      </td>
      <td align="center">
        <img src="https://github.com/HSeung03.png" width="100px;" alt="이승혁"/><br />
        <sub><b>이승혁</b></sub><br />팀원 / メンバー<br />
        <a href="https://github.com/HSeung03" target="_blank">@HSeung03</a>
      </td>
    </tr>
  </table>
</div>

---

## 📂 REPOSITORY 구조도 | リポジトリ構成図

<img src="https://github.com/HyoChan1117/HyoChan1117/raw/master/team_project-structure.drawio.png" alt="Project Structure" width="80%">

---

## 📁 디렉토리 설명 | ディレクトリの説明

| 디렉토리 | 설명 | 説明 (日本語) |
|----------|------|----------------|
| `src/` | 공통 모델 코드 및 학습/예측 모듈 | 共通のモデル・学習・予測モジュール |
| `dev/` | 팀원별 실험 공간 | メンバーごとの作業ブランチ |
| `models/` | 학습된 모델 저장 | 学習済みモデルの保存場所 |
| `outputs/` | 시각화 결과 및 평가 로그 | 可視化結果・評価ログ |
| `data/` | 공통 데이터셋 | 共通の音声データセット |
| `docker/` | Docker 실행 환경 파일 | Docker環境設定ファイル |
| `scripts/` | 유틸성 스크립트 | ユーティリティスクリプト |
| `test/` | 테스트용 오디오 샘플 | テスト用の音声ファイル |
| `.gitignore` | Git 추적 제외 설정 | Git追跡対象外ファイル設定 |

---

## ⚙️ 사용 기술 스택 | 技術スタック

| 분류 | 기술 (한국어) | 技術 (日本語) |
|------|---------------|----------------|
| 언어 | Python | Python |
| 프레임워크 | PyTorch | PyTorch |
| 도구 | Docker, GitHub Actions | Docker, GitHub Actions |
| 라이브러리 | Librosa, OpenCV, Matplotlib | Librosa, OpenCV, Matplotlib |
| 특징 추출 | MFCC, ZCR | MFCC, ZCR |
| 모델 구조 | CNN2D + LSTM | CNN2D + LSTM（時系列分類） |

---

## 🎯 프로젝트 목표 | プロジェクトの目的

| 목적 | 설명 (한국어) | 説明 (日本語) |
|------|----------------|----------------|
| 🔉 소음 분류 | 조용한 소리 / 시끄러운 소리 구분 | 静かな音と騒がしい音の分類 |
| 🎛️ 모드 전환 | 도서관 / 회의 / 쉬는 시간 등 모드별 기준 설정 | モード別（図書館・会議・休憩時間）で閾値切替 |
| 🌐 웹 시각화 | 실시간 분석 결과 시각적으로 표현 | リアルタイムに可視化表示 |
| 🧠 피드백 기반 개선 | 사용자 피드백 반영하여 모델 개선 | 利用者のフィードバックに基づいた改善 |
| 🔁 자동 측정제어 | 시간표 기반으로 자동 측정 제어 | 時間割に基づき測定を自動制御 |

---

## 🧪 프로젝트 주요 특징 | 特徴

| 항목 | 내용 (한국어) | 内容 (日本語) |
|------|------------------|------------------|
| 🎤 마이크 | 무지향성 마이크 사용 | 無指向性マイク使用 |
| 🧱 하드웨어 | Jetson Nano 기반 | Jetson Nanoによるエッジ処理 |
| 🔀 데이터
