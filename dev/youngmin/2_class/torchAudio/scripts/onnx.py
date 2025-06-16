import torch
import torch.nn as nn
from pathlib import Path

# ====== 설정 ======
MODEL_PATH = "model.pth"
ONNX_PATH = "model.onnx"
DUMMY_INPUT_SHAPE = (1, 14, 87)  # (Batch, Channels, Time) — 너가 전처리한 크기 그대로

# ====== 모델 구조 정의 (훈련 때와 동일해야 함) ======
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # input: [B, 1, 14, T]
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.fc = nn.Sequential(
            nn.Flatten(),                   # 결과: [B, 64 * 7 * (T//4)]
            nn.Linear(4032, 64),            # ← 학습 당시 입력 shape 확인해서 이 수치로 설정됨
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, 14, T]
        x = self.conv(x)
        x = self.fc(x)
        return x


# ====== 변환 ======
model = CNNClassifier()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

dummy_input = torch.randn(DUMMY_INPUT_SHAPE)  # [B, 14, 87]

Path(ONNX_PATH).parent.mkdir(parents=True, exist_ok=True)
torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=11
)

print(f"ONNX 저장 완료: {ONNX_PATH}")
