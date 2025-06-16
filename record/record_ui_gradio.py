import gradio as gr
import soundfile as sf
import pandas as pd
import os
import numpy as np
import random

# CSV 파일 불러오기
df = pd.read_csv("voice_script_100_full.csv")

# 카테고리별 문장 딕셔너리 생성
script_dict = {
    cat: df[df["category"] == cat]["sentence"].tolist()
    for cat in sorted(df["category"].unique())
}

# 전역 상태 관리
state = {
    "idx": 0,
    "speaker": "",
    "script_lines": [],
    "last_file_path": "",
}

# 녹음 파일 저장 디렉토리 설정
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RECORDINGS_DIR = os.path.join(PROJECT_ROOT, "recordings")


# 세션 시작
def start_session(speaker_name, category):
    speaker_name = speaker_name.strip()
    if not speaker_name:
        return "", "이름을 입력해주세요."

    state["speaker"] = speaker_name
    state["idx"] = 0
    state["script_lines"] = script_dict[category][:]
    random.shuffle(state["script_lines"])
    total = len(state["script_lines"])
    progress = f"1/{total} 문장 완료"

    return state["script_lines"][0], f"{speaker_name}님, '{category}' 카테고리로 시작합니다. {progress}"


# 녹음 저장
def save_recording(audio, speaker):
    if audio is None or not speaker.strip():
        return "녹음이 없거나 이름이 없습니다."

    speaker_dir = os.path.join(RECORDINGS_DIR, speaker.strip())
    os.makedirs(speaker_dir, exist_ok=True)

    filename = os.path.join(speaker_dir, f"{str(state['idx']+1).zfill(3)}.wav")
    sf.write(filename, audio[1], audio[0])
    state["last_file_path"] = filename

    total = len(state["script_lines"])
    progress = f"{state['idx']+1}/{total} 문장 완료"
    return f"저장 완료: {filename} ({progress})"


# 다음 문장으로 이동
def next_line():
    if state["idx"] + 1 < len(state["script_lines"]):
        state["idx"] += 1
        total = len(state["script_lines"])
        progress = f"{state['idx']+1}/{total} 문장 완료"
        return state["script_lines"][state["idx"]], progress
    else:
        return "모든 문장을 다 읽었습니다.", "끝!"


# 마지막 파일 삭제
def delete_last_file():
    path = state.get("last_file_path", "")
    if os.path.exists(path):
        os.remove(path)
        return f"삭제 완료: {os.path.basename(path)}"
    else:
        return "삭제할 파일이 없습니다."


# Gradio UI
with gr.Blocks() as demo:
    gr.HTML("""
    <script>
      document.addEventListener("DOMContentLoaded", function() {
        setTimeout(() => {
          document.querySelector("title").innerText = "데이터 수집 시스템 (Gradio)";
        }, 500);
      });
    </script>
    """)

    gr.Markdown("## 데이터 수집 시스템 (Gradio)")

    with gr.Row():
        name = gr.Dropdown(
            choices=["박정민", "김효찬", "배영민", "아즈키", "이현우", "이승혁"],
            label="이름을 선택하세요"
        )
        category = gr.Dropdown(choices=list(script_dict.keys()), label="카테고리 선택")

    start_btn = gr.Button("기록 시작")
    line_text = gr.Textbox(label="문장", interactive=False)
    status = gr.Textbox(label="상태", interactive=False)

    audio_input = gr.Audio(type="numpy", label="녹음하기")

    with gr.Row():
        save_btn = gr.Button("녹음 저장")
        next_btn = gr.Button("다음 문장으로")

    with gr.Row():
        playback = gr.Audio(label="최근 녹음 재생", interactive=False)
        delete_btn = gr.Button("최근 녹음 삭제")

    start_btn.click(start_session, inputs=[name, category], outputs=[line_text, status])
    save_btn.click(save_recording, inputs=[audio_input, name], outputs=status)
    next_btn.click(next_line, outputs=[line_text, status])
    save_btn.click(lambda a: a, inputs=[audio_input], outputs=playback)
    delete_btn.click(delete_last_file, outputs=status)

demo.launch()
