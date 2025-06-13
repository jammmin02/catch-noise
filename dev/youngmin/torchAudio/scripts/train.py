import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# 설정
EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 0.001
MODEL_PATH = "model.pth"

# 데이터 불러오기
x = np.load("../data/npy/x.npy")  # (N, 14, T)
y = np.load("../data/npy/y.npy")  # (N,)

x_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# 학습/검증 분할
x_train, x_val, y_train, y_val = train_test_split(x_tensor, y_tensor, test_size=0.2, random_state=42)

train_ds = TensorDataset(x_train, y_train)
val_ds = TensorDataset(x_val, y_val)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# 모델 정의
class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 2))
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (14//4) * (x.shape[2]//4), 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x):  # x: (B, 1, 14, T)
        return self.fc(self.conv(x))

# 학습 루프
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb = xb.unsqueeze(1).to(device)  # (B, 1, 14, T)
        yb = yb.to(device)

        pred = model(xb)
        loss = criterion(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"📘 Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ 모델 저장 완료: {MODEL_PATH}")
