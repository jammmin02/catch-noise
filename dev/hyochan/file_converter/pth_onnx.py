import torch
import torch.nn as nn

# CNN+LSTM 모델 정의
class CNN_LSTM(nn.Module):
    def __init__(self, input_height=86, input_width=14):
        super(CNN_LSTM, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)

        self.height = input_height // 4  # 21
        self.width = input_width // 4    # 3
        self.lstm_input_size = self.width * 64  # 192

        self.lstm = nn.LSTM(self.lstm_input_size, 64, batch_first=True)
        self.fc1 = nn.Linear(64, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        # (batch, channel=64, height=21, width=3)
        x = x.permute(0, 2, 1, 3).contiguous().view(-1, self.height, self.lstm_input_size)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x

# 모델 생성 및 weight load
model = CNN_LSTM()
model.load_state_dict(torch.load("hyochan/pytorch/pc/dataset/outputs/cnn_lstm/cnn_lstm_model.pth"))
model.eval()

# Dummy Input 생성 (실제 입력과 동일)
dummy_input = torch.randn(1, 1, 86, 14)

# ONNX 변환
torch.onnx.export(
    model,
    dummy_input,
    "hyochan/cnn_lstm_model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}},
    opset_version=13
)

print("ONNX 변환 성공!")
