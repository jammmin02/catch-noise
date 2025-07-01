import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ===== 설정 =====
X_PATH = "data/processed/x.npy"  # 전처리된 특징 벡터 (.npy)
Y_PATH = "data/processed/y.npy"  # 레이블 벡터 (.npy)
MODEL_PATH = "model/cnn_audio_classifier.pth"  # 모델 저장 경로
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 클래스 이름 매핑: 전처리와 일치해야 함
CLASS_NAMES = ['person', 'caugh', 'laugh', 'natural']

# ===== PyTorch Dataset 클래스 정의 =====
class XYDataset(Dataset):
    def __init__(self, x_path, y_path):
        # 전처리된 .npy 데이터 로딩
        self.x_data = np.load(x_path)  # shape: (N, 14, T)
        self.y_data = np.load(y_path)  # shape: (N,)
        assert self.x_data.shape[0] == self.y_data.shape[0], "데이터와 라벨 수 불일치"

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, idx):
        # 데이터 차원: (1, 14, T)로 맞추기 (채널=1)
        x = torch.tensor(self.x_data[idx], dtype=torch.float32).unsqueeze(0)
        y = int(self.y_data[idx])
        return x, y

# ===== CNN 모델 정의 =====
class SimpleCNN(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(SimpleCNN, self).__init__()

        # Convolution + Pooling 블록 2개
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # 입력 채널 1, 출력 채널 16
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 입력 채널 16, 출력 채널 32
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )

        # FC 계층 입력 차원 계산을 위한 더미 입력
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            out = self.conv(dummy)
            self.flattened_dim = out.view(1, -1).shape[1]

        # Fully Connected 계층
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

# ===== 데이터셋 및 DataLoader 준비 =====
dataset = XYDataset(X_PATH, Y_PATH)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 입력 데이터 차원 추출: (채널수, MFCC+ZCR 차원 수, 시간 축 길이)
input_shape = (1, dataset.x_data.shape[1], dataset.x_data.shape[2])
NUM_CLASSES = len(CLASS_NAMES)

# ===== 모델, 손실 함수, 옵티마이저 초기화 =====
model = SimpleCNN(input_shape=input_shape, num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ===== 학습 루프 =====
for epoch in range(EPOCHS):
    model.train()
    total_loss, correct = 0, 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        # 순전파
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 역전파 및 최적화
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 통계 집계
        total_loss += loss.item()
        correct += (outputs.argmax(1) == targets).sum().item()

    acc = correct / len(dataset)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f} - Acc: {acc:.4f}")

# ===== 학습 완료 후 모델 저장 =====
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)
print(f"모델 저장됨: {MODEL_PATH}")

# ===== 전체 데이터에 대해 혼동 행렬 평가 =====
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for inputs, targets in train_loader:
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        preds = outputs.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(targets.numpy())

# ===== 혼동 행렬 출력 및 저장 =====
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()
