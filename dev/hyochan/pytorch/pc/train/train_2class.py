import os
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ✅ MLflow 설정
now = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_name = f"{now}_hyochan_train_cnn_2class"
mlflow.set_tracking_uri("http://210.101.236.174:5000")
mlflow.set_experiment(experiment_name)
os.environ["MLFLOW_ARTIFACT_URI"] = "file:/app/mlruns"

# ✅ 경로 설정
base_dir = "hyochan/pytorch/pc/dataset/outputs/cnn_lstm"
os.makedirs(base_dir, exist_ok=True)
X_path = os.path.join(base_dir, "X_lstm.npy")
y_path = os.path.join(base_dir, "y_lstm.npy")
model_save_path = os.path.join(base_dir, "cnn_only_model.pth")
plot_save_path = os.path.join(base_dir, "train_history.png")
confusion_path = os.path.join(base_dir, "confusion_matrix.png")
confidence_plot_path = os.path.join(base_dir, "confidence_hist.png")
label_names = ['non_noisy', 'noisy']

# ✅ 데이터 로드
X = np.load(X_path)
y = np.load(y_path)
X = X[..., np.newaxis]

# ✅ 데이터 분할
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=2/9, stratify=y_temp, random_state=42)

# ✅ Torch Tensor 변환
X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2)
X_val = torch.tensor(X_val, dtype=torch.float32).permute(0, 3, 1, 2)
X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# ✅ CNN-only PyTorch 모델 정의
class CNNOnly(nn.Module):
    def __init__(self):
        super(CNNOnly, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self._get_flatten_dim(X_train), 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def _get_flatten_dim(self, sample):
        with torch.no_grad():
            x = self.pool1(self.conv1(sample[:1]))
            x = self.pool2(self.conv2(x))
            return x.numel()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x

# ✅ 학습 실행
with mlflow.start_run():
    model = CNNOnly()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    mlflow.log_params({
        "architecture": "cnn_only_pytorch",
        "optimizer": "adam",
        "loss": "binary_crossentropy",
        "batch_size": 32,
        "epochs": 30,
        "segment_duration": 3.0
    })

    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

    for epoch in range(30):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train).squeeze()
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            val_outputs = model(X_val).squeeze()
            val_loss = criterion(val_outputs, y_val)

            train_acc = ((outputs > 0.5) == y_train).float().mean().item()
            val_acc = ((val_outputs > 0.5) == y_val).float().mean().item()

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1:02d} - Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    # ✅ 테스트 평가
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).squeeze()
        y_pred_classes = (y_pred > 0.5).int()
        test_acc = (y_pred_classes == y_test.int()).float().mean().item()
        test_loss = criterion(y_pred, y_test).item()

    mlflow.log_metrics({"test_accuracy": test_acc, "test_loss": test_loss})
    print(f"Test accuracy: {test_acc:.4f} / Test loss: {test_loss:.4f}")

    # ✅ 모델 저장
    torch.save(model.state_dict(), model_save_path)
    mlflow.log_artifact(model_save_path)

    # ✅ 학습 곡선 시각화
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_accuracies, label='Train Acc', marker='o')
    plt.plot(val_accuracies, label='Val Acc', marker='x')
    plt.title('Accuracy')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Val Loss', marker='x')
    plt.title('Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_save_path)
    mlflow.log_artifact(plot_save_path)
    plt.close()

    # ✅ 혼동행렬
    cm = confusion_matrix(y_test.int(), y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(confusion_path)
    mlflow.log_artifact(confusion_path)
    plt.close()

    # ✅ Confidence 분포
    confidences = y_pred.numpy().flatten()
    plt.figure(figsize=(8, 5))
    plt.hist(confidences, bins=20, color='skyblue', edgecolor='black')
    plt.title("Confidence Distribution")
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(confidence_plot_path)
    mlflow.log_artifact(confidence_plot_path)
    plt.close()

print(f"✅ 모델 학습 및 MLflow experiment '{experiment_name}' 완료!")
