import gradio as gr
import soundfile as sf
import pandas as pd
import os
import numpy as np
import random

# スクリプトCSV読み込み
df = pd.read_csv("voice_script_100_full_jp.csv")  # <- CSV 파일명 주의

# カテゴリごとに文リストを辞書として作成
script_dict = {
    cat: df[df["category"] == cat]["sentence"].tolist()
    for cat in sorted(df["category"].unique())
}

# グローバル状態管理
state = {
    "idx": 0,
    "speaker": "",
    "script_lines": [],
    "last_file_path": "",
}

# 録音ファイル保存パス
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RECORDINGS_DIR = os.path.join(PROJECT_ROOT, "recordings")

# セッション開始処理
def start_session(speaker_name, category):
    speaker_name = speaker_name.strip()
    if not speaker_name:
        return "", "話者名を選択してください。"

    state["speaker"] = speaker_name
    state["idx"] = 0
    state["script_lines"] = script_dict[category][:]
    random.shuffle(state["script_lines"])
    total = len(state["script_lines"])
    progress = f"1/{total} 文完了"

    return state["script_lines"][0], f"{speaker_name}さん、「{category}」カテゴリで開始します。{progress}"

# 録音ファイル保存処理
def save_recording(audio, speaker):
    if audio is None or not speaker.strip():
        return "録音が存在しないか、話者名が未入力です。"

    speaker_dir = os.path.join(RECORDINGS_DIR, speaker.strip())
    os.makedirs(speaker_dir, exist_ok=True)

    filename = os.path.join(speaker_dir, f"{str(state['idx']+1).zfill(3)}.wav")
    sf.write(filename, audio[1], audio[0])
    state["last_file_path"] = filename

    total = len(state["script_lines"])
    progress = f"{state['idx']+1}/{total} 文完了"
    return f"保存完了: {filename}（{progress}）"

# 次の文へ
def next_line():
    if state["idx"] + 1 < len(state["script_lines"]):
        state["idx"] += 1
        total = len(state["script_lines"])
        progress = f"{state['idx']+1}/{total} 文完了"
        return state["script_lines"][state["idx"]], progress
    else:
        return "全ての文を読み終えました。", "終了！"

# 最新録音削除
def delete_last_file():
    path = state.get("last_file_path", "")
    if os.path.exists(path):
        os.remove(path)
        return f"削除完了: {os.path.basename(path)}"
    else:
        return "削除するファイルが存在しません。"

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 音声データ収集システム（Gradio）")

    with gr.Row():
        name = gr.Dropdown(
            choices=["박정민", "김효찬", "배영민", "아즈키", "이현우", "이승혁"],
            label="話者名を選んでください"
        )
        category = gr.Dropdown(choices=list(script_dict.keys()), label="カテゴリを選んでください")

    start_btn = gr.Button("収録開始")
    line_text = gr.Textbox(label="読み上げ文", interactive=False)
    status = gr.Textbox(label="ステータス", interactive=False)

    audio_input = gr.Audio(type="numpy", label="録音エリア")

    with gr.Row():
        save_btn = gr.Button("録音を保存")
        next_btn = gr.Button("次の文へ")

    with gr.Row():
        playback = gr.Audio(label="最新録音の再生", interactive=False)
        delete_btn = gr.Button("最新録音を削除")

    # イベント連結
    start_btn.click(start_session, inputs=[name, category], outputs=[line_text, status])
    save_btn.click(save_recording, inputs=[audio_input, name], outputs=status)
    next_btn.click(next_line, outputs=[line_text, status])
    save_btn.click(lambda a: a, inputs=[audio_input], outputs=playback)
    delete_btn.click(delete_last_file, outputs=status)

# アプリケーション起動
demo.launch()
