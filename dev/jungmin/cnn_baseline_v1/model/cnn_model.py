import torch 
import torch.nn as nn

class CNNClassifier(nn.Module):
    """
    입력 특징맵을 받아 다중 클래스 분류를 수행하는 CNN 분류기
    """
    def __init__(self, input_channels, num_classes=5):
        super(CNNClassifier, self).__init__()

        # CNN 레이어를 순차적으로 구성
        self.net = nn.Sequential(
            # Conv 블록 1: Conv → BN → ReLU → MaxPool → Dropout
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),

            # Conv 블록 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),

            # Conv 블록 3 + AdaptiveAvgPool (출력을 1x1로 축소)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),

            # Fully connected layers
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)  # 최종 클래스 수만큼 출력
        )

    def forward(self, x):
        """
        순전파
        """
        return self.net(x)
