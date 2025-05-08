import os
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ✅ 환경 설정
now = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_name = f"train_cnn_lstm_2class_hyochan_{now}"

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
mlflow.set_tracking_uri("http://210.101.236.174:5000")
mlflow.set_experiment(experiment_name)

# 💡 artifact 경로 명시 (이게 없으면 오류남!)
os.environ["MLFLOW_ARTIFACT_URI"] = "file:/app/mlruns"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 모델 정의
class CNN_LSTM(nn.Module):
    def __init__(self, input_shape):
        super(CNN_LSTM, self).__init__()
        C, H, W = input_shape
        self.conv1 = nn.Conv2d(C, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        conv_out_h, conv_out_w = H // 4, W // 4
        self.lstm_input_size = conv_out_w * 64
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=64, batch_first=True)
        self.fc1 = nn.Linear(64, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        B, C, H, W = x.size()
        x = x.view(B, H, -1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x

# ✅ 학습 실행
with mlflow.start_run():
    base_dir = "hyochan/pytorch_2class/dataset/outputs/cnn_lstm"
    os.makedirs(base_dir, exist_ok=True)

    # 📁 데이터 로드
    X = np.load(os.path.join(base_dir, "X_lstm.npy"))[..., np.newaxis]
    y = np.load(os.path.join(base_dir, "y_lstm.npy"))
    X = np.transpose(X, (0, 3, 1, 2))  # (B, C, H, W)

    # 📊 분할
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=2/9, stratify=y_temp, random_state=42)

    def to_loader(X, y, batch_size=32, shuffle=True):
        return DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)),
                          batch_size=batch_size, shuffle=shuffle)

    train_loader = to_loader(X_train, y_train)
    val_loader = to_loader(X_val, y_val, shuffle=False)
    test_loader = to_loader(X_test, y_test, shuffle=False)

    # 🧠 모델 초기화
    model = CNN_LSTM((1, X.shape[2], X.shape[3])).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 🧾 파라미터 로깅
    mlflow.log_params({
        "segment_duration": 3.0,
        "optimizer": "adam",
        "loss": "binary_crossentropy",
        "batch_size": 32,
        "epochs": 30,
        "architecture": "cnn_lstm"
    })

    # 🔁 학습 루프
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_loss = float("inf")
    patience, patience_limit = 0, 5
    for epoch in range(30):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            correct += ((pred > 0.5).float() == yb).sum().item()
            total += xb.size(0)
        train_losses.append(total_loss / total)
        train_accs.append(correct / total)

        # 🔍 Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * xb.size(0)
                val_correct += ((pred > 0.5).float() == yb).sum().item()
                val_total += xb.size(0)
        val_losses.append(val_loss / val_total)
        val_accs.append(val_correct / val_total)

        print(f"Epoch {epoch+1}: Train Loss={train_losses[-1]:.4f}, Val Loss={val_losses[-1]:.4f}")

        # ⛔ Early stopping
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            patience = 0
            best_model = model.state_dict()
        else:
            patience += 1
            if patience >= patience_limit:
                print("⛔ Early stopping")
                break

    model.load_state_dict(best_model)

    # 🧪 테스트
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            all_preds.extend(pred)
            all_labels.extend(yb.numpy())
    y_pred = np.array(all_preds).flatten()
    y_true = np.array(all_labels)
    y_pred_classes = (y_pred > 0.5).astype(int)

    test_acc = accuracy_score(y_true, y_pred_classes)
    test_loss = criterion(torch.tensor(y_pred).unsqueeze(1), torch.tensor(y_true).unsqueeze(1)).item()
    mlflow.log_metrics({"test_accuracy": test_acc, "test_loss": test_loss})

    # 💾 모델 저장
    model_path = os.path.join(base_dir, "cnn_lstm_model.pt")
    torch.save(model.state_dict(), model_path)
    mlflow.log_artifact(model_path)

    # 📈 시각화
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.legend(), plt.title("Accuracy")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend(), plt.title("Loss")
    plt.grid(True)
    plt.tight_layout()
    history_path = os.path.join(base_dir, "train_history.png")
    plt.savefig(history_path)
    mlflow.log_artifact(history_path)
    plt.close()

    # 📊 혼동 행렬
    cm = confusion_matrix(y_true, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["non_noisy", "noisy"])
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(base_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)
    plt.close()

    # 📈 confidence 분포
    plt.figure(figsize=(8, 5))
    plt.hist(y_pred, bins=20, color='skyblue', edgecolor='black')
    plt.title("Confidence Distribution")
    plt.xlabel("Confidence"), plt.ylabel("Frequency")
    plt.grid(True), plt.tight_layout()
    conf_path = os.path.join(base_dir, "confidence_hist.png")
    plt.savefig(conf_path)
    mlflow.log_artifact(conf_path)
    plt.close()

print(f"✅ MLflow experiment '{experiment_name}' 등록 완료!")
