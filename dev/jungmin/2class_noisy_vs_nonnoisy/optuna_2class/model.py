import torch
import torch.nn as nn

class CNNLSTM(nn.Module):
    def __init__(self, conv1_filters, conv2_filters, lstm_units, dense_units, dropout):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, conv1_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # ↓ H, W 절반
            nn.Conv2d(conv1_filters, conv2_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)   # ↓ 또 절반
        )
        self.lstm_input_channels = conv2_filters
        self.lstm = nn.LSTM(input_size=self.lstm_input_channels, hidden_size=lstm_units, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(lstm_units, dense_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_units, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        x = self.cnn(x)  # → (B, C, H, W)

        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)     # → (B, H, W, C)
        x = x.reshape(B, H * W, C)    # → (B, T, C)

        x, _ = self.lstm(x)           # → (B, T, lstm_units)
        x = x[:, -1, :]               # → (B, lstm_units)
        return self.fc(x)             # → (B, 1)
