import torch
import torch.nn as nn

class CNNOnly(nn.Module):
    """
    CNN 기반의 2-class 오디오 분류 모델

    - LSTM 제거 (실시간 최적화 및 TensorRT 변환 용이)
    - 입력: (batch_size, 1, height, width)
    - 출력: 이진 분류 logits (Sigmoid 미적용 상태)
    """
    def __init__(self, input_shape, conv1_filters, conv2_filters, dense_units, dropout):
        """
        Parameters
        ----------
        input_shape : tuple
            입력 데이터의 (height, width), ex: (86, 14)
        conv1_filters : int
            첫번째 Conv2D filter 수
        conv2_filters : int
            두번째 Conv2D filter 수
        dense_units : int
            Dense layer의 hidden unit 수
        dropout : float
            Dropout 비율
        """
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, conv1_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(conv1_filters, conv2_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Conv 연산 후 feature map 차원 계산 (TensorRT 변환 위해 고정 필수)
        height, width = input_shape
        height = height // 4  # MaxPool2d(2) 두 번 적용 → 1/4 축소
        width = width // 4
        self.flatten_dim = conv2_filters * height * width

        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_dim, dense_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_units, 1)
            # Sigmoid 제거 → BCEWithLogitsLoss()에서 내부 적용 예정
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
            출력 logits, shape: (batch_size, 1)
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        return logits
