import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error # MSE


# 난수 생성
# supppress=True : 아주 작은 숫자를 0으로 보이게함
# precision=1 : 소수점 1자리까지만 출력
# X  : 0 ~ 10사이 난수 100개 (100행 1열)
np.set_printoptions(suppress=True, precision=1)
X = np.random.rand(3, 1) * 10 

# 타겟값 y 생성
# randn 정규분포 노이즈 (주로 -3~3, 여기에 2를 곱해서 스케일 업)
y = 2.5 * X + np.random.randn(3, 1) * 2
y = y.ravel() # 다차원 배열을 1차원으로 평탄화
# SGD는 1차원 타겟값을 요구구s

# 모델 생성, 학습, 예측, 평가
model = SGDRegressor(
    max_iter=100, # 학습 반복 횟수 
    learning_rate="adaptive", # 학습률 자동조절
    eta0=0.001, # 학습률 값 # 0은 초기값을 말함, constant는 의미X
    penalty=None, # 정규화 제거
    random_state=0 # 결과 재현을 위한 시드 고정
)
model.fit(X, y)
y_pred = model.predict(X) # 예측 값 출력 (일반화)
mse = mean_squared_error(y, y_pred)
print(f"평균 제곱 오차(MSE): {mse:.4f}")
