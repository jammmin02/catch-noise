import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================
# 설정
# ============================================
X_PATH = Path("/workspace/data/x.npy")
Y_PATH = Path("/workspace/data/y.npy")
MODEL_PATH = Path("/workspace/data/only-cnn.pth")

BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.001

# ============================================
# 모델 정의
# ============================================
class CNNOnly(nn.Module):
    def __init__(self):
        super(CNNOnly, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 2))
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 21 * 3, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ============================================
# 학습 루틴
# ============================================
if __name__ == "__main__":
    # 데이터 로딩
    x = np.load(X_PATH)
    y = np.load(Y_PATH).astype(np.float32)
    x = torch.tensor(x, dtype=torch.float32).permute(0, 3, 1, 2)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    # 데이터 분할
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3, random_state=42)

    # 오버샘플링을 위한 샘플러 생성
    class_sample_counts = torch.bincount(y_train.squeeze().long())
    weights = 1.0 / class_sample_counts.float()
    sample_weights = weights[y_train.squeeze().long()]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    # 데이터로더 구성
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=BATCH_SIZE)

    # 모델 & 학습 준비
    model = CNNOnly()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses, val_losses, val_accs = [], [], []

    # 학습 루프
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 검증
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        preds_all = []
        labels_all = []
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item()
                pred_label = (torch.sigmoid(pred) > 0.5).float()
                correct += (pred_label == yb).sum().item()
                total += yb.size(0)
                preds_all.extend(pred_label.cpu().numpy())
                labels_all.extend(yb.cpu().numpy())

        acc = correct / total
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(acc)

        print(f"[{epoch+1}/{EPOCHS}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {acc:.2f}")

    # ============================================
    # 결과 저장
    # ============================================
    # 모델 저장
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"모델 저장 완료: {MODEL_PATH.name}")

    # 손실 그래프
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.savefig("/workspace/data/loss_curve.png")
    plt.close()

    # 정확도 그래프
    plt.figure()
    plt.plot(val_accs, label="Val Accuracy", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.ylim([0, 1.05])
    plt.legend()
    plt.savefig("/workspace/data/acc_curve.png")
    plt.close()

    # 혼동 행렬
    cm = confusion_matrix(labels_all, preds_all)
    disp = ConfusionMatrixDisplay(cm, display_labels=["non_noisy", "noisy"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig("/workspace/data/confusion_matrix.png")
    plt.close()