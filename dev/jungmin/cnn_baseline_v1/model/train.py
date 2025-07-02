import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import mlflow

# CNNClassifier 강화 버전
class CNNClassifier(nn.Module):
    def __init__(self, input_channels=1, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# Dataset 클래스
class FeatureDataset(Dataset):
    def __init__(self, data_dir, df, label_map):
        self.data_dir = data_dir
        self.df = df
        self.label_map = label_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feature_path = os.path.join(self.data_dir, row["filename"].replace(".wav", ".npy"))
        feature = np.load(feature_path)
        feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)
        label = self.label_map[row["label"]]
        return feature, label

# 그래프 분리 출력
def plot_separate_metrics(train_losses, train_accs, val_losses, val_accs, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, train_accs, label="Train Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Train Metrics")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "train_metrics.png"))
    plt.close()

    plt.figure()
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.plot(epochs, val_accs, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Validation Metrics")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "val_metrics.png"))
    plt.close()

# 혼동행렬 출력
def evaluate_and_confusion(model, loader, device, label_map, save_path):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            outputs = model(X)
            _, preds = torch.max(outputs, 1)
            y_true.extend(y.numpy())
            y_pred.extend(preds.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=list(label_map.keys()))
    disp.plot(cmap="Blues")
    plt.title("Test Confusion Matrix")
    plt.savefig(save_path)
    plt.close()

def main():
    mlflow.set_tracking_uri("http://mlflow:5000")

    with mlflow.start_run(run_name="cnn_with_dropout_batchnorm_scheduler"):
        # 하이퍼파라미터
        batch_size = 32
        lr = 1e-3
        epochs = 20
        weight_decay = 1e-4
        label_map = {"coughs": 0, "laugh": 1, "natural": 2, "person": 3}

        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("weight_decay", weight_decay)

        # 데이터셋 분할
        df = pd.read_csv("./dataset/labels.csv")
        train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=1/3, stratify=temp_df["label"], random_state=42)

        train_ds = FeatureDataset("./dataset/processed", train_df, label_map)
        val_ds = FeatureDataset("./dataset/processed", val_df, label_map)
        test_ds = FeatureDataset("./dataset/processed", test_df, label_map)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CNNClassifier(input_channels=1, num_classes=4).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

        train_losses, val_losses, train_accs, val_accs = [], [], [], []

        for epoch in range(epochs):
            # 학습
            model.train()
            total_loss, correct, total = 0, 0, 0
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                outputs = model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * X.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == y).sum().item()
                total += y.size(0)

            avg_loss = total_loss / total
            acc = correct / total

            # 검증
            model.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(device), y.to(device)
                    outputs = model(X)
                    loss = criterion(outputs, y)

                    val_loss += loss.item() * X.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == y).sum().item()
                    val_total += y.size(0)

            avg_val_loss = val_loss / val_total
            val_acc = val_correct / val_total

            train_losses.append(avg_loss)
            train_accs.append(acc)
            val_losses.append(avg_val_loss)
            val_accs.append(val_acc)

            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            mlflow.log_metric("train_acc", acc, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)

            scheduler.step(avg_val_loss)

            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f} Acc: {acc:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.4f}")

        # 그래프
        plot_separate_metrics(train_losses, train_accs, val_losses, val_accs, "./outputs/visualizations")
        mlflow.log_artifact("./outputs/visualizations/train_metrics.png")
        mlflow.log_artifact("./outputs/visualizations/val_metrics.png")

        # 혼동행렬
        evaluate_and_confusion(model, test_loader, device, label_map, "./outputs/visualizations/test_confusion.png")
        mlflow.log_artifact("./outputs/visualizations/test_confusion.png")

        # 모델 저장
        torch.save(model.state_dict(), "cnn_model.pt")
        mlflow.log_artifact("cnn_model.pt")

if __name__ == "__main__":
    main()
