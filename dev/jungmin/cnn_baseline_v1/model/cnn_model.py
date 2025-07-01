# Placeholder
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNClassifier(nn.Module):
    def __init__(self, input_channels, num_classes=5):
        """
        CNN 분류기 초기화
        - input_channels: 입력 채널 수 (예: MFCC+ZCR+RMS 차원)
        - num_classes: 출력 클래스 수 (기본 5개 클래스)
        """
        super(CNNClassifier, self).__init__()

        # 첫 번째 convolution 블록: Conv → BN → MaxPool
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=(3,3), padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d((2,2))

        # 두 번째 convolution 블록
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3,3), padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d((2,2))

        # 세 번째 convolution 블록 + Adaptive Pool (출력을 1x1로 축소)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3,3), padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.AdaptiveAvgPool2d((1,1))

        # 최종 fully connected layer (64 → num_classes)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        """
        순전파 정의
        - 입력 x: (batch_size, channels, feature_dim, time_steps)
        - 출력: softmax 확률 분포
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = x.view(x.size(0), -1)  # 1차원으로 평탄화
        x = self.fc(x)

        # softmax로 클래스별 확률 출력
        x = F.softmax(x, dim=1)
        return x
