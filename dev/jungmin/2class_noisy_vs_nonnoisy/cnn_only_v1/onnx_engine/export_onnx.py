import os
import json
import torch
import onnx
from model import CNNOnly

# 기본 경로 (cnn_only_v2 기준으로 통일)
BASE_DIR = "dev/jungmin/2class_noisy_vs_nonnoisy/cnn_only_v2"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# 하이퍼파라미터 및 경로 설정
param_path = os.path.join(OUTPUT_DIR, "best_params.json")
model_path = os.path.join(OUTPUT_DIR, "best_model.pt")
onnx_path = os.path.join(OUTPUT_DIR, "best_model.onnx")

# 하이퍼파라미터 불러오기
with open(param_path, "r") as f:
    best_params = json.load(f)

# 모델 정의 (input_shape은 전처리 기준 고정)
input_shape = (86, 14)
model = CNNOnly(
    input_shape=input_shape,
    conv1_filters=best_params["conv1_filters"],
    conv2_filters=best_params["conv2_filters"],
    dense_units=best_params["dense_units"],
    dropout=best_params["dropout"]
).to("cpu")

# 학습된 모델 파라미터 로드
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# 더미 입력 (TensorRT 변환을 위한 입력 샘플)
dummy_input = torch.randn(1, 1, input_shape[0], input_shape[1])
print(f"Dummy input shape: {dummy_input.shape}")

# ONNX 변환 실행
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

# 변환된 ONNX 모델 검증
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
print("ONNX 모델 검증 정상 완료")
