import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ===== 설정 =====
X_PATH = "data/processed/x.npy"
Y_PATH = "data/processed/y.npy"
MODEL_PATH = "model/cnn_audio_classifier.pth"
CLASS_NAMES = ['person', 'cough', 'natural']
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Dataset 정의 =====
class XYDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x = x_data  # shape: (N, 14, T)
        self.y = y_data

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx], dtype=torch.float32).unsqueeze(0)  # (1, 14, T)
        y = int(self.y[idx])
        return x, y

# ===== 모델 정의 =====
class SimpleCNN(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            out = self.conv(dummy)
            self.flattened_dim = out.view(1, -1).shape[1]
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

# ===== 데이터 로딩 및 분할 =====
x_all = np.load(X_PATH)
y_all = np.load(Y_PATH)
x_train, x_test, y_train, y_test = train_test_split(
    x_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

train_loader = DataLoader(XYDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(XYDataset(x_test, y_test), batch_size=BATCH_SIZE)

input_shape = (1, x_all.shape[1], x_all.shape[2])
model = SimpleCNN(input_shape=input_shape, num_classes=len(CLASS_NAMES)).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ===== 학습 루프 =====
for epoch in range(EPOCHS):
    model.train()
    total_loss, correct = 0, 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        preds = model(x_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (preds.argmax(1) == y_batch).sum().item()
    acc = correct / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f} - Acc: {acc:.4f}")

# ===== 모델 저장 =====
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)
print("모델 저장 완료:", MODEL_PATH)

# ===== 혼동 행렬 평가 (테스트셋) =====
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(DEVICE)
        preds = model(x_batch).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.numpy())

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix (Test Set)")
plt.savefig("confusion_matrix.png")
plt.show()
