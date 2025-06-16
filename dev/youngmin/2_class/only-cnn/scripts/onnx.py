import torch
from train import CNNOnly  # 동일한 모델 클래스 import
from pathlib import Path

# ============================================
# 경로 설정
# ============================================

MODEL_PATH = Path("/workspace/data/only-cnn.pth")
ONNX_PATH = Path("/workspace/data/only-cnn.onnx")

# ONNX 변환 시 더미 입력 (CNN 입력 형태)
DUMMY_INPUT = torch.randn(1, 1, 86, 13)

# ============================================
# 모델 로드 및 변환
# ============================================

# 모델 로딩
model = CNNOnly()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# ONNX로 변환
torch.onnx.export(
    model,
    DUMMY_INPUT,
    ONNX_PATH,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11
)

print(f"ONNX 변환 완료: {ONNX_PATH.name}")
