import torch
import onnx
import json
import os
from model import CNNOnly

# 기본 경로 (cnn_only_v2 기준)
BASE_DIR = "dev/jungmin/2class_noisy_vs_nonnoisy/cnn_only_v2"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# 최적 하이퍼파라미터 로드
param_path = os.path.join(OUTPUT_DIR, "best_params.json")
with open(param_path, "r") as f:
    best_params = json.load(f)

# CNN-only 모델 정의 (input_shape 명확히 반영)
input_shape = (86, 14)  # (height, width) 전처리 기준 유지
model = CNNOnly(
    input_shape=input_shape,
    conv1_filters=best_params["conv1_filters"],
    conv2_filters=best_params["conv2_filters"],
    dense_units=best_params["dense_units"],
    dropout=best_params["dropout"]
).to("cpu")

# 학습된 가중치 로드
model_path = os.path.join(OUTPUT_DIR, "best_model.pt")
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# 더미 입력 준비 (batch, channel, height, width)
dummy_input = torch.randn(1, 1, input_shape[0], input_shape[1])
print(f"Dummy input shape: {dummy_input.shape}")
assert dummy_input.shape == (1, 1, 86, 14), f"입력 shape 불일치: {dummy_input.shape}"

# ONNX 변환 수행
onnx_path = os.path.join(OUTPUT_DIR, "best_model.onnx")
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11
)
print(f"ONNX 변환 완료 → {onnx_path}")

# ONNX 모델 검증
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
print("ONNX 모델 검증 완료 - 정상")
