import torch
import torch.nn as nn

class CNNOnly(nn.Module):
    """
    CNN 기반의 2-class 오디오 분류 모델 (LSTM 제거)

    - 실시간 추론 및 TensorRT 변환에 최적화된 CNN-only 구조
    - 입력 shape: (batch_size, 1, height, width)
    - 출력: 이진 분류 logits (Sigmoid 미적용 → BCEWithLogitsLoss 사용 예정)
    """

    def __init__(self, input_shape, conv1_filters, conv2_filters, dense_units, dropout):
        """
        모델 초기화

        Parameters
        ----------
        input_shape : tuple
            입력 데이터의 (height, width). 
            예: robust_v7 기준 (max_len, feature_dim)
        conv1_filters : int
            첫 번째 Conv2D layer의 filter 개수
        conv2_filters : int
            두 번째 Conv2D layer의 filter 개수
        dense_units : int
            Fully Connected layer의 hidden unit 수
        dropout : float
            Dropout 비율 (0.0 ~ 0.5 권장)
        """
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, conv1_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(conv1_filters, conv2_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # Conv 연산 후 출력 feature map 크기 계산
        height, width = input_shape  # ex: (max_len, feature_dim)
        height = height // 4  # MaxPool2d(2) 두 번 → 1/4 축소
        width = width // 4
        self.flatten_dim = conv2_filters * height * width

        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_dim, dense_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_units, 1)  # 최종 1차원 logit 출력
        )

    def forward(self, x):
        """
        Forward propagation

        Parameters
        ----------
        x : torch.Tensor
            입력 tensor, shape: (batch_size, 1, height, width)

        Returns
        -------
        logits : torch.Tensor
            출력 logits (BCEWithLogitsLoss에서 직접 사용)
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        logits = self.classifier(x)
        return logits
