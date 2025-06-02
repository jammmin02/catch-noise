import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

def stratified_split(X, y, seed=42):
    """
    주어진 데이터(X, y)를 클래스 비율을 유지하며
    학습:검증:테스트 = 7:2:1 비율로 분할하는 함수

    Parameters
    ----------
    X : np.ndarray
        입력 데이터 (샘플 수, 시간, 특징)
    y : np.ndarray
        라벨 데이터 (샘플 수, )
    seed : int
        랜덤 시드 값

    Returns
    -------
    (X_train, y_train), (X_val, y_val), (X_test, y_test) : tuple
        분할된 학습, 검증, 테스트 데이터셋
    """
    X_train_all, X_val_all, X_test_all = [], [], []
    y_train_all, y_val_all, y_test_all = [], [], []

    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        X_cls, y_cls = X[idx], y[idx]

        # 1단계: 전체 10%를 테스트 세트로 분리
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_cls, y_cls, test_size=0.1, random_state=seed, shuffle=True
        )

        # 2단계: 나머지 90%를 7:2 비율로 학습/검증 세트 분할
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=2/9, random_state=seed, shuffle=True
        )

        X_train_all.append(X_train)
        y_train_all.append(y_train)
        X_val_all.append(X_val)
        y_val_all.append(y_val)
        X_test_all.append(X_test)
        y_test_all.append(y_test)

    def concat_and_shuffle(X_parts, y_parts, seed=42):
        """
        분할된 데이터를 하나로 합치고 섞어주는 함수
        """
        X_all = np.concatenate(X_parts)
        y_all = np.concatenate(y_parts)
        np.random.seed(seed)
        idx = np.random.permutation(len(X_all))
        return X_all[idx], y_all[idx]

    X_train, y_train = concat_and_shuffle(X_train_all, y_train_all, seed=seed)
    X_val, y_val     = concat_and_shuffle(X_val_all, y_val_all, seed=seed)
    X_test, y_test   = concat_and_shuffle(X_test_all, y_test_all, seed=seed)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def load_data(batch_size, base_dir=None):
    """
    저장된 .npy 파일을 로드하고, stratified split을 통해
    학습, 검증, 테스트 DataLoader를 생성하는 함수

    Parameters
    ----------
    batch_size : int
        DataLoader에 사용할 배치 사이즈
    base_dir : str
        데이터가 저장된 경로 (default: 프로젝트 디렉토리 하드코딩)

    Returns
    -------
    train_loader, val_loader, test_loader : tuple
        학습, 검증, 테스트용 DataLoader
    """
    if base_dir is None:
        base_dir = "/app/dev/jungmin/2class_noisy_vs_nonnoisy/cnn_only_v1/outputs"

    X = np.load(os.path.join(base_dir, "X_cnn.npy"))  # 입력 특징 (샘플 수, 시간, 특징)
    y = np.load(os.path.join(base_dir, "y_cnn.npy"))  # 라벨 (샘플 수, )

    # Stratified 7:2:1 split
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = stratified_split(X, y)

    def to_tensor(X, y):
        """
        Numpy 배열을 TensorDataset으로 변환하는 함수

        CNN 모델 입력 형태를 위해 채널 차원(1) 추가
        라벨은 BCEWithLogitsLoss를 위해 float32 + unsqueeze 처리
        """
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # [B, 1, 시간, 특징]
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # [B, 1]
        return TensorDataset(X_tensor, y_tensor)

    train_dataset = to_tensor(X_train, y_train)
    val_dataset   = to_tensor(X_val, y_val)
    test_dataset  = to_tensor(X_test, y_test)

    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader