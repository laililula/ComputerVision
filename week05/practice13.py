# TensorFlow 라이브러리를 tf라는 이름으로 불러옵니다.
# TensorFlow는 딥러닝 모델을 만들고 학습시키는 핵심 라이브러리입니다.
import tensorflow as tf

# TensorFlow 안에 포함된 고수준 딥러닝 API인 Keras를 불러옵니다.
# keras는 신경망 모델을 더 쉽게 만들 수 있게 도와줍니다.
from tensorflow import keras

# 신경망의 각 층(Layer)을 만들기 위해 layers 모듈을 불러옵니다.
from tensorflow.keras import layers

# 수치 계산과 배열 처리를 위한 NumPy를 불러옵니다.
import numpy as np

# 그래프를 그리기 위한 matplotlib를 불러옵니다.
import matplotlib.pyplot as plt


# MNIST 데이터셋을 불러옵니다.
# x_train, y_train: 학습용 이미지와 정답
# x_test, y_test: 테스트용 이미지와 정답
# MNIST는 손글씨 숫자(0~9) 이미지 데이터셋입니다.
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


# 이미지 데이터의 자료형을 float32로 바꾸고 255로 나누어 0~1 범위로 정규화합니다.
# 원래 픽셀 값은 0~255인데, 이렇게 범위를 줄이면 학습이 더 안정적으로 진행됩니다.
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


# MNIST 이미지는 원래 (28, 28) 형태의 2차원 배열입니다.
# Dense(완전연결층)는 1차원 벡터 입력을 받기 때문에,
# 각 이미지를 28*28 = 784 길이의 1차원 벡터로 펼칩니다.
# -1은 "데이터 개수는 자동으로 맞춰라"는 뜻입니다.
x_train_flat = x_train.reshape(-1, 28 * 28)
x_test_flat = x_test.reshape(-1, 28 * 28)


# 순차적으로 층을 쌓는 Sequential 모델을 생성합니다.
# 이 모델은 입력층 -> 은닉층1 -> 은닉층2 -> 출력층 순서로 구성됩니다.
model = keras.Sequential([
    
    # 입력 데이터의 형태를 지정합니다.
    # 각 입력 샘플은 길이 784의 벡터여야 한다는 뜻입니다.
    layers.Input(shape=(28 * 28,)),
    
    # 첫 번째 은닉층입니다.
    # 뉴런 128개를 가지며, 활성화 함수로 ReLU를 사용합니다.
    # ReLU는 음수는 0으로, 양수는 그대로 통과시켜 비선형성을 부여합니다.
    layers.Dense(128, activation='relu'),
    
    # 두 번째 은닉층입니다.
    # 첫 번째 층에서 추출한 특징을 더 정교하게 가공합니다.
    layers.Dense(64, activation='relu'),
    
    # 출력층입니다.
    # 숫자 0~9를 분류해야 하므로 뉴런 10개를 둡니다.
    # softmax는 각 숫자일 확률을 출력하며, 전체 합은 1이 됩니다.
    layers.Dense(10, activation='softmax')
])


# 모델의 학습 방법을 설정합니다.
model.compile(
    # adam: 자주 쓰이는 최적화 알고리즘으로, 가중치를 효율적으로 업데이트합니다.
    optimizer='adam',
    
    # 정답이 one-hot 벡터가 아니라 정수 라벨(예: 3, 7)이므로
    # sparse_categorical_crossentropy를 사용합니다.
    loss='sparse_categorical_crossentropy',
    
    # 학습 중 정확도도 함께 확인합니다.
    metrics=['accuracy']
)


# 모델 구조를 출력합니다.
# 각 층의 이름, 출력 형태, 파라미터 수를 확인할 수 있습니다.
print("모델 아키텍처:")
model.summary()
print()


# 모델 학습 시작
print("모델 훈련 중...")

# model.fit()은 실제로 학습을 수행하는 부분입니다.
history = model.fit(
    # 입력 데이터
    x_train_flat,
    
    # 정답 라벨
    y_train,
    
    # 전체 학습 데이터를 10번 반복 학습합니다.
    epochs=10,
    
    # 한 번에 128개씩 데이터를 묶어서 학습합니다.
    # 너무 작으면 느리고, 너무 크면 메모리를 많이 씁니다.
    batch_size=128,
    
    # 학습 데이터의 10%를 검증용으로 자동 분리합니다.
    # 검증 데이터는 학습에는 사용하지 않고 성능 확인용으로만 사용합니다.
    validation_split=0.1,
    
    # 학습 진행 상황을 화면에 출력합니다.
    verbose=1
)
print()


# 학습이 끝난 뒤, 한 번도 학습에 사용하지 않은 테스트 데이터로 성능을 평가합니다.
print("테스트 세트에서 성능 평가:")
test_loss, test_accuracy = model.evaluate(x_test_flat, y_test, verbose=0)

# 정확도는 0~1 사이 값이므로 100을 곱해 퍼센트로 출력합니다.
print(f"테스트 정확도: {test_accuracy * 100:.2f}%")

# loss는 모델의 오차 정도를 나타냅니다. 일반적으로 낮을수록 좋습니다.
print(f"테스트 손실(Loss): {test_loss:.4f}")
print()


# 학습 과정에서 정확도와 손실이 어떻게 변했는지 시각화합니다.
plt.figure(figsize=(12, 4))


# 첫 번째 그래프 위치(1행 2열 중 1번째)
plt.subplot(1, 2, 1)

# history.history['accuracy']는 각 epoch마다의 학습 정확도입니다.
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)

# history.history['val_accuracy']는 각 epoch마다의 검증 정확도입니다.
# 학습 정확도와 검증 정확도를 비교하면 과적합 여부를 어느 정도 볼 수 있습니다.
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)


# 두 번째 그래프 위치(1행 2열 중 2번째)
plt.subplot(1, 2, 2)

# 학습 손실 변화
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)

# 검증 손실 변화
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# 그래프들이 겹치지 않도록 자동으로 여백을 조정합니다.
plt.tight_layout()

# 그래프를 화면에 출력합니다.
plt.show()


# 테스트 데이터 일부에 대해 실제 예측 결과를 확인합니다.
print("테스트 이미지에 대한 예측:")

# 처음 10개의 테스트 이미지만 확인합니다.
num_samples = 10

# model.predict()는 각 이미지가 0~9일 확률을 출력합니다.
# 예를 들어 [0.01, 0.02, 0.90, ...] 같은 형태입니다.
predictions = model.predict(x_test_flat[:num_samples])


# 예측 결과를 이미지와 함께 시각화합니다.
plt.figure(figsize=(15, 3))

for i in range(num_samples):
    # 2행 5열 구조로 이미지를 하나씩 배치합니다.
    plt.subplot(2, 5, i + 1)
    
    # 원래 2차원 이미지 형태의 테스트 이미지를 출력합니다.
    plt.imshow(x_test[i], cmap='gray')
    
    # np.argmax()는 가장 확률이 높은 클래스의 인덱스를 반환합니다.
    # 즉, 모델이 예측한 숫자입니다.
    predicted_label = np.argmax(predictions[i])
    
    # 실제 정답 라벨입니다.
    true_label = y_test[i]
    
    # 예측이 맞으면 초록색, 틀리면 빨간색으로 제목을 표시합니다.
    color = 'green' if predicted_label == true_label else 'red'
    
    # 예측값과 실제값을 제목에 표시합니다.
    plt.title(f'Pred: {predicted_label}\nTrue: {true_label}', color=color)
    
    # 축은 필요 없으므로 숨깁니다.
    plt.axis('off')

plt.tight_layout()
plt.show()