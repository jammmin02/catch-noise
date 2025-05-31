import torch
import torch.nn as nn

class CNNLSTM(nn.Module):
    def __init__(self, conv1_filters, conv2_filters, lstm_units, dense_units, dropout):
        super().__init__()

        # ✅ CNN 계층: 특징 추출
        self.cnn = nn.Sequential(
            nn.Conv2d(1, conv1_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(conv1_filters, conv2_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # ✅ LSTM 계층 (항상 dropout=0 → cuDNN 영향 제거)
        self.lstm = nn.LSTM(
            input_size=conv2_filters,   # CNN 출력 채널 수
            hidden_size=lstm_units,
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )

        # ✅ FC 계층: 이진 분류
        self.fc = nn.Sequential(
            nn.Linear(lstm_units, dense_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_units, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 입력 x: (B, 1, H, W)
        x = self.cnn(x)                     # → (B, C, H', W')
        B, C, H, W = x.size()
        x = x.permute(0, 2, 3, 1)           # → (B, H', W', C)
        x = x.reshape(B, H * W, C)          # → (B, T, C)
        lstm_out, _ = self.lstm(x)          # → (B, T, lstm_units)
        x = lstm_out[:, -1, :]              # 마지막 타임스텝
        return self.fc(x)                   # → (B, 1)
