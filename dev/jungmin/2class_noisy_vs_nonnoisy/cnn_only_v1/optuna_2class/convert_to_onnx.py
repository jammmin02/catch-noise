import os
import json
import torch
import numpy as np

from model import CNNOnly

# 기본 경로 설정
base_dir = "/app/dev/jungmin/2class_noisy_vs_nonnoisy/cnn_only_v1/outputs"

# Optuna best params 로드
with open(os.path.join(base_dir, "best_params.json"), "r") as f:
    best_params = json.load(f)

# 입력 shape 자동 감지 (robust_v7 기준)
X_sample = np.load(os.path.join(base_dir, "X_cnn.npy"))
input_shape = X_sample.shape[1:]  # (time_steps, n_features)

# CNN 모델 생성 (학습과 동일한 구조)
model = CNNOnly(
    input_shape=input_shape,
    conv1_filters=best_params["conv1_filters"],
    conv2_filters=best_params["conv2_filters"],
    dense_units=best_params["dense_units"],
    dropout=best_params["dropout"]
)

# 학습 완료 모델 로드
model_path = os.path.join(base_dir, "best_model.pt")
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# 더미 입력 준비 (ONNX 변환용)
dummy_input = torch.randn(1, 1, *input_shape)  # (batch, channel, height, width)

# 변환 경로
onnx_path = os.path.join(base_dir, "best_model.onnx")

# ONNX 변환 (static shape, opset 17 사용 권장 → TensorRT 안정성 ↑)
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output']
)

print(f"ONNX 변환 완료 → {onnx_path}")
