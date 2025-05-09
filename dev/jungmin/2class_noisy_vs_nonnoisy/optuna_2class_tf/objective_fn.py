import mlflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Reshape, LSTM, Dense, Dropout
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping

def build_model(input_shape, trial, timesteps, features):
    """
    🎯 Optuna로부터 하이퍼파라미터를 받아 모델을 생성합니다.
    """
    model = Sequential()
    model.add(Conv2D(trial.suggest_categorical("conv1_filters", [16, 32, 64]),
                     (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(trial.suggest_categorical("conv2_filters", [32, 64, 128]),
                     (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Reshape((timesteps, features)))

    model.add(LSTM(trial.suggest_int("lstm_units", 32, 128, step=32)))

    model.add(Dense(trial.suggest_int("dense_units", 32, 128, step=32), activation='relu'))

    model.add(Dropout(trial.suggest_float("dropout", 0.2, 0.5, step=0.1)))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=trial.suggest_float("lr", 1e-4, 1e-2, log=True)),
        loss=BinaryCrossentropy(),
        metrics=['accuracy']
    )
    return model

def objective(trial, X_train, X_val, y_train, y_val, timesteps, features):
    """
    🧪 Optuna가 호출하는 objective 함수: 모델 구성 + 학습 + 검증
    """
    with mlflow.start_run(nested=True):  # 각 trial마다 run 남기기
        input_shape = (X_train.shape[1], X_train.shape[2], 1)

        # 모델 생성
        model = build_model(input_shape, trial, timesteps, features)

        # 조기 종료 콜백
        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        # 학습
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=30,
            batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]),
            callbacks=[early_stop],
            verbose=0
        )

        # 검증 성능
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)

        # 📜 하이퍼파라미터 및 결과 기록
        mlflow.log_params(trial.params)
        mlflow.log_metrics({"val_loss": val_loss, "val_accuracy": val_acc})

        return val_loss  # 최소화 목표 (또는 -val_accuracy 사용 가능)
