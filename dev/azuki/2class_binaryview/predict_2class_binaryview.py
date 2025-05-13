# 📦 라이브러리 불러오기
import os
import subprocess
import uuid
import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from tensorflow.keras.models import load_model

# 🔧 설정값
sr = 22050
n_mfcc = 13
hop_length = 512
segment_duration = 5.0
frame_per_second = sr / hop_length
max_len = int(frame_per_second * segment_duration)

# 📁 경로
model_path = "outputs/cnn_lstm_bview_model.h5"
test_folder = "test_audio_batch"
visual_dir = os.path.join("outputs", "visuals_bview")
matrix_path = os.path.join("outputs", "confusion_matrix.png")

# 🏷️ 클래스
class_names = ['quiet', 'loud']
class_colors = {'quiet': 'skyblue', 'loud': 'tomato'}

# 모델 로드
model = load_model(model_path)

# 🔄 오디오 파일 변환 함수
def convert_to_wav(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.wav':
        return file_path, False
    temp_wav = f"temp_{uuid.uuid4().hex[:8]}.wav"
    command = ['ffmpeg', '-y', '-i', file_path, '-ac', '1', '-ar', str(sr), temp_wav]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return temp_wav, True

# 🔍 특징 추출 함수
def preprocess_segment(y_audio):
    if np.max(np.abs(y_audio)) < 1e-4:
        print("⚠️ [무음] 감지됨")
        return None
    mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(y=y_audio, hop_length=hop_length)
    features = np.vstack([mfcc, zcr])
    if features.shape[1] < max_len:
        features = np.pad(features, ((0, 0), (0, max_len - features.shape[1])), mode='constant')
    else:
        features = features[:, :max_len]
    return features.T[np.newaxis, ..., np.newaxis]

# 🎯 세그먼트 단위 예측
def predict_file(file_path):
    y_full, _ = librosa.load(file_path, sr=sr)
    duration = librosa.get_duration(y=y_full, sr=sr)
    segments = int(np.ceil(duration / segment_duration))
    results = []
    for i in range(segments):
        start = int(i * segment_duration * sr)
        end = int(min((i + 1) * segment_duration * sr, len(y_full)))
        segment = y_full[start:end]
        if len(segment) < sr:
            continue
        x = preprocess_segment(segment)
        if x is None:
            continue
        pred = model.predict(x, verbose=0)[0]
        label = 'loud' if pred[0] > 0.5 else 'quiet'  # ← ここが修正点！
        results.append((f"seg{i+1}", pred, label))
    return results

# 📊 시각화 및 혼동행렬
def plot_results(results, true_labels, pred_labels):
    if not results:
        print("❗ 예측 결과 없음.")
        return

    os.makedirs(visual_dir, exist_ok=True)

    acc = accuracy_score(true_labels, pred_labels)
    print(f"\n✅ 전체 정확도: {acc * 100:.2f}% ({sum(np.array(true_labels)==np.array(pred_labels))}/{len(true_labels)})")
    print("📊 예측된 클래스 수:", dict(Counter(pred_labels)))

    def chunk_list(data, chunk_size=100):
        for i in range(0, len(data), chunk_size):
            yield data[i:i+chunk_size]

    results.sort(key=lambda x: np.max(x[1]), reverse=True)

    for idx, chunk in enumerate(chunk_list(results), start=1):
        names = [x[0] for x in chunk]
        probs = [np.max(x[1]) for x in chunk]
        labels = [x[2] for x in chunk]
        colors = [class_colors[label] for label in labels]

        plt.figure(figsize=(14, max(5, len(names)*0.4)))
        sns.set(style="whitegrid")
        bars = plt.barh(names, probs, color=colors, edgecolor='black')
        plt.xlabel("예측 확률 (Softmax)", fontsize=13)
        plt.title(f"세그먼트 예측 결과 (Part {idx})", fontsize=16)
        plt.xlim(0, 1)
        plt.gca().invert_yaxis()

        for bar, prob, label in zip(bars, probs, labels):
            view_label = "🔊 loud" if label == "loud" else "🤫 quiet"
            plt.text(prob + 0.01, bar.get_y() + bar.get_height()/2,
                     f"{prob:.2f} ({view_label})", va='center', fontsize=10)

        plt.tight_layout()
        fname = os.path.join(visual_dir, f"all_segments_result_{idx}.png")
        plt.savefig(fname)
        plt.close()

    cm = confusion_matrix(true_labels, pred_labels, labels=class_names)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    plt.figure(figsize=(5.5, 4.5))
    disp.plot(cmap='Blues', values_format='d')
    plt.title("혼동 행렬 (True vs Predicted)")
    plt.tight_layout()
    plt.savefig(matrix_path)
    plt.close()

# 🚀 실행
if __name__ == "__main__":
    results_all = []
    true_labels = []
    pred_labels = []

    if not os.path.exists(test_folder):
        print(f"❗ 폴더 '{test_folder}'가 존재하지 않습니다.")
    else:
        files = []
        for sub in os.listdir(test_folder):
            sub_path = os.path.join(test_folder, sub)
            if os.path.isdir(sub_path):
                files += [os.path.join(sub_path, f) for f in os.listdir(sub_path) if f.lower().endswith(('.wav', '.mp4', '.m4a', '.mp3'))]

        for fpath in files:
            fname = os.path.basename(fpath)
            fpath, is_temp = convert_to_wav(fpath)

            # ✅ 폴더 이름에서 정답 라벨을 가져오기
            true_label = os.path.basename(os.path.dirname(fpath)).lower()
            if true_label not in class_names:
                true_label = "unknown"

            results = predict_file(fpath)
            for r in results:
                segment_name = f"{fname} - {r[0]}"
                results_all.append((segment_name, r[1], r[2]))
                true_labels.append(true_label)
                pred_labels.append(r[2])

            if is_temp and os.path.exists(fpath):
                os.remove(fpath)

        plot_results(results_all, true_labels, pred_labels)
