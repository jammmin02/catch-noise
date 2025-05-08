import torch
import os
import numpy as np
from model import CNNLSTM
from data_loader import load_data
import mlflow
from torch.nn import BCELoss
from torch.optim import Adam
from sklearn.metrics import accuracy_score

best_params = {
    "conv1_filters": 32,
    "conv2_filters": 64,
    "lstm_units": 64,
    "dense_units": 64,
    "dropout": 0.3,
    "lr": 0.001,
    "batch_size": 32
}

# ✅ MLflow 설정
mlflow.set_tracking_uri("http://210.101.236.174:5000")
mlflow.set_experiment("optuna_cnn_lstm_2class")

with mlflow.start_run(run_name="train_best_pytorch"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ 데이터 로드
    X_train, _, X_test, _, _ = load_data(best_params["batch_size"])  # timesteps, features 제거

    # ✅ 모델 생성 (timesteps, features 전달 X)
    model = CNNLSTM(
        conv1_filters=best_params["conv1_filters"],
        conv2_filters=best_params["conv2_filters"],
        lstm_units=best_params["lstm_units"],
        dense_units=best_params["dense_units"],
        dropout=best_params["dropout"]
    ).to(device)

    loss_fn = BCELoss()
    optimizer = Adam(model.parameters(), lr=best_params["lr"])

    for epoch in range(15):
        model.train()
        for xb, yb in X_train:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = loss_fn(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # ✅ 평가
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in X_test:
            xb = xb.to(device)
            output = model(xb)
            y_pred += (output.cpu().numpy() > 0.5).astype(int).tolist()
            y_true += yb.numpy().astype(int).tolist()

    acc = accuracy_score(y_true, y_pred)
    print(f"✅ Test Accuracy: {acc:.4f}")
    mlflow.log_metric("test_accuracy", acc)
    mlflow.log_params(best_params)

    # ✅ 저장
    model_dir = "outputs/cnn_lstm"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "best_model.pt")
    torch.save(model.state_dict(), model_path)
    mlflow.log_artifact(model_path)
    print(f"📦 모델 저장 완료: {model_path}")
