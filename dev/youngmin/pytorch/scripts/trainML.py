import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from pathlib import Path
import mlflow
import mlflow.pytorch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import os



# 1. 모델 정의
class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(4032, 64)  # 여기 수정!
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.pool1(self.conv1(x)))
        x = torch.relu(self.pool2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x


# 2. 데이터 불러오기
x = np.load("../data/x.npy")  # shape: (9374, 86, 14)
y = np.load("../data/y.npy")  # shape: (9374,)

# 🔸 float32 변환 (PyTorch는 float32 권장)
x = x.astype(np.float32)
y = y.astype(np.float32)

# 🔸 채널 차원 추가 (PyTorch용: (N, 1, H, W))
x = np.expand_dims(x, axis=1)  # shape: (9374, 1, 86, 14)

# 🔸 표준 정규화 (z-score)
x = (x - x.mean()) / x.std()

# 🔸 학습/검증 분리
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# 🔸 텐서로 변환
train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
val_dataset = TensorDataset(torch.tensor(x_val), torch.tensor(y_val))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)


# 3. 학습 준비
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AudioCNN().to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 학습 루프
best_val_loss = float('inf')
patience, wait = 5, 0

for epoch in range(30):
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
            preds = model(xb)
            val_loss += criterion(preds, yb).item()

    avg_train_loss = running_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)

    print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        wait = 0
        torch.save(model.state_dict(), "../model/cnn_model.pt")
        print(" 모델 저장됨")
    else:
        wait += 1
        if wait >= patience:
            print(" Early Stopping")
            break
# MLflow 설정
mlflow.set_tracking_uri("http://210.101.236.174:5000")
mlflow.set_experiment("Classroom Noise Detection")

with mlflow.start_run(run_name="Baeyoungmin"):

    mlflow.log_param("batch_size", 32)
    mlflow.log_param("epochs", 30)
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("loss_function", "BCELoss")

    # 최종 모델 저장
    mlflow.pytorch.log_model(model, "model")

    # 검증용 예측 및 혼동행렬
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            preds = model(xb).cpu().numpy()
            all_preds.extend((preds > 0.5).astype(int).flatten())
            all_labels.extend(yb.numpy().astype(int))

    acc = accuracy_score(all_labels, all_preds)
    mlflow.log_metric("val_accuracy", acc)

    # 혼동 행렬 저장 및 로깅
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["quiet", "loud"])
    disp.plot()
    plt.title("Confusion Matrix")
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)
    plt.close()

    # 손실값 시각화 (학습 과정에서 수집 필요)
    train_losses = []  # 루프에서 매 epoch마다 append 필요
    val_losses = []    # 루프에서 매 epoch마다 append 필요

    # ... 위 학습 루프 내에서
    # train_losses.append(avg_train_loss)
    # val_losses.append(avg_val_loss)

    # 그래프 저장 및 로깅
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training & Validation Loss")
    loss_path = "loss_graph.png"
    plt.savefig(loss_path)
    mlflow.log_artifact(loss_path)
    plt.close()