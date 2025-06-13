import torch
import torch.nn as nn

class CNNOnly(nn.Module):
    def __init__(self, conv1_filters, conv2_filters, dense_units, dropout):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, conv1_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(conv1_filters, conv2_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # 🔽 input feature 수를 동적으로 계산 (forward에서 결정)
        self.flatten_dim = None

        # 분리: Linear는 forward에서 동적 생성
        self.dense_units = dense_units
        self.dropout = dropout
        self.conv2_filters = conv2_filters

    def forward(self, x):
        x = self.features(x)
        B = x.size(0)
        x = x.view(B, -1)  # flatten

        # 최초 forward 시에만 dense layer 생성 (lazy init)
        if self.flatten_dim is None:
            self.flatten_dim = x.size(1)
            self.classifier = nn.Sequential(
                nn.Linear(self.flatten_dim, self.dense_units),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.dense_units, 2)  # 🔥 2-class 분류!
            ).to(x.device)  # device 동기화

        return self.classifier(x)
