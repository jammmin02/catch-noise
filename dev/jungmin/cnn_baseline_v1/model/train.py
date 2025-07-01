import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import mlflow
from cnn_model import CNNClassifier

class FeatureDataset(Dataset):
    """
    전처리된 특징 벡터(.npy)와 라벨을 로드하는 Dataset 클래스
    """
    def __init__(self, data_dir, df, label_map):
        self.data_dir = data_dir  # 특징 벡터 파일 경로
        self.df = df              # 라벨 데이터프레임
        self.label_map = label_map  # 클래스명 → 인덱스 매핑

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        한 샘플의 특징 벡터와 라벨 반환
        - 특징: (1, feature_dim, time_steps) 형태
        - 라벨: int index
        """
        row = self.df.iloc[idx]
        feature_path = os.path.join(self.data_dir, row["filename"].replace(".wav", ".npy"))
        feature = np.load(feature_path)
        feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)  # 채널 차원 추가
        label = self.label_map[row["label"]]
        return feature, label

def plot_metrics(losses, val_losses, accs, val_accs, save_path):
    """
    학습/검증 손실 및 정확도를 그래프로 그려 파일로 저장
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(losses) + 1)

    plt.figure()
    plt.plot(epochs, losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.plot(epochs, accs, label="Train Acc")
    plt.plot(epochs, val_accs, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.title("Training/Validation Metrics")
    plt.savefig(save_path)
    plt.close()

def main():
    """
    CNN 학습 및 검증 루프 + 메트릭 시각화 + MLflow 로깅
    """
    mlflow.set_tracking_uri("http://mlflow:5000")

    with mlflow.start_run(run_name="cnn_baseline_train"):
        # 하이퍼파라미터 및 라벨 매핑 정의
        batch_size = 32
        lr = 1e-3
        epochs = 20
        input_channels = 1
        label_map = {"coughs": 0, "laugh": 1, "natural": 2, "person": 3}

        # MLflow 파라미터 기록
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("label_map", label_map)

        # 데이터 로드 및 train/val/test 분할
        df = pd.read_csv("./dataset/labels.csv")
        train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=1/3, stratify=temp_df["label"], random_state=42)

        # Dataset 및 DataLoader 생성
        train_ds = FeatureDataset("./dataset/processed", train_df, label_map)
        val_ds = FeatureDataset("./dataset/processed", val_df, label_map)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # 모델 초기화 및 학습 설정
        device = torch.device("cpu")
        model = CNNClassifier(input_channels=input_channels).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # 메트릭 기록용 리스트
        train_losses, val_losses, train_accs, val_accs = [], [], [], []

        # 학습 루프
        for epoch in range(epochs):
            # 학습 단계
            model.train()
            total_loss, correct, total = 0, 0, 0

            for X, y in train_loader:
                X = X.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                outputs = model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

                # 손실 및 정확도 집계
                total_loss += loss.item() * X.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == y).sum().item()
                total += y.size(0)

            avg_loss = total_loss / total
            acc = correct / total

            # 검증 단계
            model.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            with torch.no_grad():
                for X, y in val_loader:
                    X = X.to(device)
                    y = y.to(device)

                    outputs = model(X)
                    loss = criterion(outputs, y)

                    val_loss += loss.item() * X.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == y).sum().item()
                    val_total += y.size(0)

            avg_val_loss = val_loss / val_total
            val_acc = val_correct / val_total

            # 메트릭 기록
            train_losses.append(avg_loss)
            val_losses.append(avg_val_loss)
            train_accs.append(acc)
            val_accs.append(val_acc)

            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            mlflow.log_metric("train_acc", acc, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)

            # epoch별 출력
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {avg_loss:.4f} Acc: {acc:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.4f}")

        # 학습/검증 메트릭 시각화 및 저장
        plot_path = "./outputs/visualizations/train_val_metrics.png"
        plot_metrics(train_losses, val_losses, train_accs, val_accs, plot_path)
        mlflow.log_artifact(plot_path)

        # 학습된 모델 저장 및 artifact 등록
        torch.save(model.state_dict(), "cnn_model.pt")
        mlflow.log_artifact("cnn_model.pt")

if __name__ == "__main__":
    main()