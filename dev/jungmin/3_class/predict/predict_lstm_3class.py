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

# 🔧 Config
sr = 22050
n_mfcc = 13
hop_length = 512
segment_duration = 3.0
frame_per_second = sr / hop_length
max_len = int(frame_per_second * segment_duration)

model_path = "dev/jungmin/3_class/dataset/outputs/cnn_lstm/cnn_lstm_model.h5"
test_folder = "dev/jungmin/test_audio_batch"
save_dir = os.path.join(test_folder, "visuals")

class_names = ['silent', 'neutral', 'noisy']
class_colors = {'silent': 'skyblue', 'neutral': 'orange', 'noisy': 'tomato'}

# 📥 Load model
model = load_model(model_path)

# 🎧 Convert to WAV if needed
def convert_to_wav(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.wav':
        return file_path
    temp_wav = f"temp_{uuid.uuid4().hex[:8]}.wav"
    command = ['ffmpeg', '-y', '-i', file_path, '-ac', '1', '-ar', str(sr), temp_wav]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return temp_wav

# 🎛️ Feature extraction
def preprocess_segment(y_audio):
    mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(y=y_audio, hop_length=hop_length)
    features = np.vstack([mfcc, zcr])
    if features.shape[1] < max_len:
        features = np.pad(features, ((0, 0), (0, max_len - features.shape[1])), mode='constant')
    else:
        features = features[:, :max_len]
    return features.T[np.newaxis, ..., np.newaxis]  # (1, max_len, 14, 1)

# 🔍 Predict per segment
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
        pred = model.predict(x, verbose=0)[0]
        label_idx = np.argmax(pred)
        label = class_names[label_idx]
        results.append((f"seg{i+1}", pred, label))
    return results

# 📊 Plot results
def plot_results(results, true_labels, pred_labels):
    if not results:
        print("❗ No prediction results found.")
        return

    os.makedirs(save_dir, exist_ok=True)

    # Accuracy
    acc = accuracy_score(true_labels, pred_labels)
    print(f"\n✅ Overall Accuracy: {acc * 100:.2f}% ({sum(np.array(true_labels)==np.array(pred_labels))}/{len(true_labels)})")

    # Class count
    pred_counts = Counter(pred_labels)
    print("📊 Prediction counts:", dict(pred_counts))

    # 🔁 Split into chunks of 100
    def chunk_list(data, chunk_size=100):
        for i in range(0, len(data), chunk_size):
            yield data[i:i+chunk_size]

    results.sort(key=lambda x: np.max(x[1]), reverse=True)

    # 🔄 For each chunk of 100 segments
    for idx, chunk in enumerate(chunk_list(results, 100), start=1):
        names = [x[0] for x in chunk]
        probs = [np.max(x[1]) for x in chunk]
        labels = [x[2] for x in chunk]
        colors = [class_colors[label] for label in labels]

        plt.figure(figsize=(14, max(5, len(names)*0.4)))
        sns.set(style="whitegrid")
        bars = plt.barh(names, probs, color=colors, edgecolor='black')
        plt.xlabel("Prediction Probability (Max Softmax)", fontsize=13)
        plt.title(f"Segment Predictions (Part {idx})", fontsize=16)
        plt.xlim(0, 1)
        plt.gca().invert_yaxis()
        for bar, prob, label in zip(bars, probs, labels):
            plt.text(prob + 0.01, bar.get_y() + bar.get_height()/2,
                     f"{prob:.2f} ({label})", va='center', fontsize=10)
        plt.tight_layout()
        fname = os.path.join(save_dir, f"all_segments_result_{idx}.png")
        plt.savefig(fname)
        plt.close()  # 💡 메모리 절약

    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=class_names)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    plt.figure(figsize=(5.5, 4.5))
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Confusion Matrix (True vs Predicted)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()


# 🚀 Run predictions
if __name__ == "__main__":
    results_all = []
    true_labels = []
    pred_labels = []

    if not os.path.exists(test_folder):
        print(f"❗ Folder '{test_folder}' does not exist.")
    else:
        for fname in os.listdir(test_folder):
            if not fname.lower().endswith(('.wav', '.mp4', '.m4a', '.mp3')):
                continue
            fpath = os.path.join(test_folder, fname)
            if not fname.endswith('.wav'):
                fpath = convert_to_wav(fpath)

            matched = [c for c in class_names if c in fname.lower()]
            true_label = matched[0] if matched else "unknown"

            results = predict_file(fpath)
            for r in results:
                segment_name = f"{fname} - {r[0]}"
                results_all.append((segment_name, r[1], r[2]))
                true_labels.append(true_label)
                pred_labels.append(r[2])

        plot_results(results_all, true_labels, pred_labels)