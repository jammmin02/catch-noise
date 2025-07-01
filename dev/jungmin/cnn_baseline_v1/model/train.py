import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import mlflow
from cnn_model import CNNClassifier
import pandas as pd

class FeatureDataset(Dataset):
    """
    전처리된 특징 벡터(.npy)와 라벨을 로드하는 Dataset 클래스
    """
    def __init__(self, data_dir, label_csv, label_map):
        self.data_dir = data_dir
        self.df = pd.read_csv(label_csv)
        self.label_map = label_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # .npy 특징 로드 및 torch tensor 변환 (채널 차원 추가)
        feature = np.load(os.path.join(self.data_dir, row["filename"].replace(".wav", ".npy")))
        feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)  # (1, feat, time)
        label = self.label_map[row["label"]]
        return feature, label

def main():
    """
    CNN 분류기 학습 루프
    - DataLoader, 모델, 손실함수, 옵티마이저 설정
    - epoch 단위 학습 및 MLflow 로깅
    """
    mlflow.set_tracking_uri("http://mlflow:5000")
    with mlflow.start_run(run_name="cnn_baseline_train"):
        # 하이퍼파라미터 설정
        batch_size = 32
        lr = 1e-3
        epochs = 20
        input_channels = 1

        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("epochs", epochs)

        # 데이터셋 및 DataLoader 준비
        label_map = {"voice":0, "machine":1, "ambient":2, "cough":3, "movement":4}
        dataset = FeatureDataset("./dataset/processed", "./dataset/labels.csv", label_map)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 모델 및 학습 설정
        model = CNNClassifier(input_channels=input_channels)
        model = model.cuda()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # 학습 루프
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            correct = 0
            total = 0

            for X, y in loader:
                X = X.cuda()
                y = y.cuda()

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

            # MLflow 메트릭 기록
            mlflow.log_metric("loss", avg_loss, step=epoch)
            mlflow.log_metric("accuracy", acc, step=epoch)

            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Acc: {acc:.4f}")

        # 학습된 모델 저장 및 MLflow artifact 등록
        torch.save(model.state_dict(), "cnn_model.pt")
        mlflow.log_artifact("cnn_model.pt")

if __name__ == "__main__":
    main()
