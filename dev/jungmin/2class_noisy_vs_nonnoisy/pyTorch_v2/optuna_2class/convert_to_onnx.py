import torch
import onnx
import json
from model import CNNLSTM

# ✅ 1. best_params.json 로드
param_path = "outputs/cnn_lstm/best_params.json"
with open(param_path, "r") as f:
    best_params = json.load(f)

# ✅ 2. 모델 정의 (best trial 기준)
model = CNNLSTM(
    conv1_filters=best_params["conv1_filters"],
    conv2_filters=best_params["conv2_filters"],
    lstm_units=best_params["lstm_units"],
    dense_units=best_params["dense_units"],
    dropout=best_params["dropout"]
).to("cpu")

# ✅ 3. 학습된 가중치 로드
model.load_state_dict(torch.load("outputs/cnn_lstm/best_model.pt", map_location="cpu"))
model.eval()

# ✅ 4. 더미 입력 준비 (입력 shape은 [B, C=1, H=86, W=14])
dummy_input = torch.randn(1, 1, 86, 14)
print("🧪 dummy_input shape:", dummy_input.shape)
assert dummy_input.dim() == 4, f"❌ 입력 텐서가 4차원이 아닙니다: {dummy_input.shape}"

# ✅ 5. ONNX 변환
onnx_path = "outputs/cnn_lstm/best_model.onnx"
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
