import torch
import onnx
import json
import os
from model import CNNOnly  # CNN-only v2 구조에서 이 모델 작성된다고 가정

# ✅ 경로 통일 (cnn_only_v2 기준)
BASE_DIR = "dev/jungmin/2class_noisy_vs_nonnoisy/cnn_only_v2"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# ✅ 1. best_params.json 로드
param_path = os.path.join(OUTPUT_DIR, "best_params.json")
with open(param_path, "r") as f:
    best_params = json.load(f)

# ✅ 2. 모델 정의 (best trial 기준)
model = CNNOnly(
    conv1_filters=best_params["conv1_filters"],
    conv2_filters=best_params["conv2_filters"],
    dense_units=best_params["dense_units"],
    dropout=best_params["dropout"]
).to("cpu")

# ✅ 3. 학습된 가중치 로드
model_path = os.path.join(OUTPUT_DIR, "best_model.pt")
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# ✅ 4. 더미 입력 준비 (CNN-only는 일반적으로 [B, 1, H, W])
# ✅ 전처리 max_len, n_mfcc 참고해서 dummy shape 설정
n_mfcc = 13
max_len = 86  # 이 값은 실제 전처리 summary.json에서 불러와도 좋음

dummy_input = torch.randn(1, 1, max_len, n_mfcc)  # (batch, channel, height, width)
print("🧪 dummy_input shape:", dummy_input.shape)
assert dummy_input.dim() == 4, f"❌ 입력 텐서가 4차원이 아닙니다: {dummy_input.shape}"

# ✅ 5. ONNX 변환
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
print(f"✅ PyTorch → ONNX 변환 완료: {onnx_path}")

# ✅ 6. ONNX 모델 검증
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
print("✅ ONNX 모델 검증 완료!")
