import torch
import torch.nn as nn
import numpy as np

# 모델 클래스 정의 (train.py와 동일해야 함)
class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(4032, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.pool1(self.conv1(x)))
        x = torch.relu(self.pool2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

# 모델 불러오기
model = AudioCNN()
model.load_state_dict(torch.load("../model/cnn_model.pt", map_location="cpu"))
model.eval()

# 더미 입력 (실제 입력과 같은 shape)
dummy_input = torch.randn(1, 1, 86, 14)  # (Batch=1, Channels=1, Height=86, Width=14)

# ONNX로 내보내기
torch.onnx.export(
    model,
    dummy_input,
    "../model/cnn_model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11
)

print("모델이 성공적으로 model/cnn_model.onnx 파일로 변환되었습니다.")
