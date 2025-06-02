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

        # 🔽 input feature 수를 동적으로 계산
        self.flatten_dim = None  # 처음에 None으로 두고 forward에서 계산

        self.classifier_head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_units, 1),
            nn.Sigmoid()
        )

        self.dense_units = dense_units
        self.dropout = dropout
        self.conv2_filters = conv2_filters

    def forward(self, x):
      x = self.features(x)
      B = x.size(0)
      x = x.view(B, -1)  # flatten

      if self.flatten_dim is None:
        self.flatten_dim = x.size(1)
        # Linear 계층 정의 + 자동 device 적용
        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_dim, self.dense_units),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.dense_units, 1),
            nn.Sigmoid()
        ).to(x.device)  # 🔥 중요: 이걸로 GPU에 자동 이동됨

      return self.classifier(x)