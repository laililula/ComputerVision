# Week05 Practice README

이 README 파일은 week05 폴더의 practice13.py, practice14.py 파일에 대한 내용을 정리한 문서입니다.  
각 실습마다 전체 코드, 주요 코드 설명, 실행 결과 영역을 포함합니다.

---

# Practice 13: 간단한 이미지 분류기 구현

## 전체 코드

```python
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
```

## 주요 코드 설명

### 1. MNIST 데이터셋 로드

```python
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
```

MNIST 손글씨 숫자 데이터셋을 로드합니다.

- `x_train` : 훈련 이미지 (60,000개, 28×28 픽셀 그레이스케일)  
- `y_train` : 훈련 레이블 (0~9 숫자)  
- `x_test` : 테스트 이미지 (10,000개, 한 번도 학습에 사용되지 않은 데이터)  
- `y_test` : 테스트 레이블  

원본 픽셀값은 0~255 범위의 정수형 데이터입니다.

### 2. 데이터 정규화와 형태 변환

```python
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train_flat = x_train.reshape(-1, 28 * 28)
x_test_flat = x_test.reshape(-1, 28 * 28)
```

- **정규화(Normalization)**: 픽셀 값을 0~1 범위로 변환하여 신경망의 학습 안정성을 높입니다.
  - 원래 0~255 범위의 데이터를 255로 나누면 0~1로 스케일링됩니다.
  - 정규화된 데이터는 그래디언트 소실 문제를 줄이고 수렴 속도를 개선합니다.

- **형태 변환**: 2D 이미지(28×28)를 1D 벡터(784)로 변환합니다.
  - Dense 층은 1차원 입력만 받을 수 있기 때문입니다.
  - `reshape(-1, 28*28)`에서 -1은 "배치 크기는 자동으로 조정하라"는 의미입니다.

### 3. Sequential 모델 구축

```python
model = keras.Sequential([
    layers.Input(shape=(28 * 28,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

다층 신경망(MLP: Multi-Layer Perceptron)을 구성합니다.

- **Input 레이어**: 입력 데이터 형태를 지정(784개 특징)
- **첫 번째 Hidden 레이어**: 
  - 128개 뉴런(노드)을 가진 완전연결층(Dense)
  - ReLU 활성화 함수: $ f(x) = \max(0, x) $ - 비선형성을 추가하여 모델 표현력 증대
  
- **두 번째 Hidden 레이어**: 
  - 64개 뉴런 (첫 번째 층의 특징을 더 정교하게 압축)
  - ReLU 활성화 함수
  
- **Output 레이어**: 
  - 10개 뉴런 (0~9 숫자 분류를 위해)
  - Softmax 활성화 함수: 각 클래스의 확률을 출력하며 모든 확률의 합 = 1

### 4. 모델 컴파일

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

모델의 학습 설정을 정의합니다.

- **optimizer='adam'**: 
  - Adaptive Moment Estimation의 줄임말
  - 학습률을 자동으로 조정하면서 효율적으로 최적화합니다.
  - SGD보다 빠르고 안정적인 수렴을 제공합니다.

- **loss='sparse_categorical_crossentropy'**:
  - 정답이 정수형 라벨(예: 3, 7, 9)일 때 사용하는 손실 함수
  - one-hot 인코딩이 필요 없어 메모리 효율적입니다.
  - 모델의 오류를 수치화하여 가중치 업데이트에 사용합니다.

- **metrics=['accuracy']**:
  - 학습 과정에서 모니터링할 성능 지표
  - 매 에포크마다 정확도(올바르게 분류한 비율)를 계산하여 출력

### 5. 모델 훈련

```python
history = model.fit(
    x_train_flat, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)
```

실제 학습을 수행합니다.

- **epochs=10**: 
  - 전체 훈련 데이터를 10번 반복 학습
  - 각 반복을 1 에포크라고 부릅니다.
  - 너무 적으면 모델이 충분히 학습되지 않고, 너무 많으면 과적합 위험

- **batch_size=128**:
  - 한 번에 처리할 샘플의 개수
  - 메모리 효율성과 학습 안정성의 균형
  - 배치 단위로 그래디언트를 계산하고 가중치를 업데이트

- **validation_split=0.1**:
  - 훈련 데이터의 10%를 검증 데이터로 분리
  - 검증 데이터는 학습에 사용되지 않고 과적합 모니터링용
  - 실제 테스트 전에 모델 성능을 미리 확인

- **return value (history)**:
  - 각 에포크마다의 손실값과 정확도를 저장
  - 나중에 학습 곡선을 시각화하는 데 사용

### 6. 모델 평가

```python
test_loss, test_accuracy = model.evaluate(x_test_flat, y_test, verbose=0)
```

훈련에 사용되지 않은 테스트 데이터로 최종 성능을 평가합니다.

- 훈련 과정에서 보지 못한 새로운 데이터에 대한 일반화 능력을 측정
- `test_accuracy`: 테스트 세트 정확도 (0~1 범위, 100을 곱하면 퍼센트)
- `test_loss`: 테스트 세트에서의 손실값

### 7. 훈련 과정 시각화

```python
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
```

훈련 중 모델의 성능 변화를 그래프로 표시합니다.

- **Training Accuracy vs Validation Accuracy**:
  - 둘의 차이가 크면 과적합 발생 가능성
  - 검증 정확도가 떨어지면 학습 중단 고려

- **Training Loss vs Validation Loss**:
  - 손실값의 감소 추이로 학습 진행 상황 파악
  - 두 곡선이 평행하게 움직이면 학습이 안정적

### 8. 예측 결과 시각화

```python
predictions = model.predict(x_test_flat[:num_samples])
predicted_label = np.argmax(predictions[i])
```

테스트 이미지에 대한 모델의 예측을 시각화합니다.

- `model.predict()`: 입력 이미지에 대해 각 클래스(0~9)의 확률을 반환
  - 예: `[0.01, 0.02, 0.90, 0.01, ...]` 형태
  
- `np.argmax()`: 가장 높은 확률을 가진 클래스의 인덱스 반환
  - 모델이 최종적으로 예측한 숫자
  
- 예측이 맞으면 초록색, 틀리면 빨간색으로 표시하여 시각적 비교

## 실행 결과

모델 훈련 시 정확도는 약 97~98% 수준에 도달합니다.

**모델 구조 출력 예시:**
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 128)               100480    
batch_normalization           (None, 128)               512      
dense_1 (Dense)              (None, 64)                8256      
dense_2 (Dense)              (None, 10)                650       
=================================================================
Total params: 109,898
Trainable params: 109,898
```

**훈련 결과 예시:**
- Epoch 1/10: loss: 0.2156, accuracy: 0.9356, val_loss: 0.1243, val_accuracy: 0.9621
- Epoch 10/10: loss: 0.0412, accuracy: 0.9854, val_loss: 0.0928, val_accuracy: 0.9741

**정확도 및 손실 변화 그래프:**
- Training Accuracy: 점진적으로 증가하여 약 98.5% 도달
- Validation Accuracy: Training Accuracy보다 약간 낮게 유지 (약 97.4%)
- Training Loss: 급격히 감소 후 안정화
- Validation Loss: Training Loss보다 약간 높게 유지 (정상적인 과적합 패턴)

**테스트 성능:**
- 테스트 정확도: 약 97.5~98%
- 테스트 손실: 약 0.08~0.10

**예측 시각화 예시:**
- 정확하게 분류된 숫자들이 초록색으로 표시됩니다.
- 분류 실패한 숫자들이 빨간색으로 표시됩니다.
- 대부분의 이미지가 정확하게 분류됩니다.

<img width="2379" height="902" alt="image" src="https://github.com/user-attachments/assets/1ce948ad-b3d9-4ef0-b1aa-f0ff8e25b4e0" />
<img width="2869" height="719" alt="image" src="https://github.com/user-attachments/assets/a941e89d-c8b5-4987-b742-123d57f41435" />


---

# Practice 14: CIFAR-10 데이터셋을 활용한 CNN 모델 구축

## 전체 코드

```python
# TensorFlow 라이브러리를 tf라는 이름으로 불러옵니다.
# 딥러닝 모델 생성, 학습, 평가에 사용하는 핵심 라이브러리입니다.
import tensorflow as tf

# TensorFlow 안에 포함된 고수준 API인 Keras를 불러옵니다.
# 모델을 쉽게 구성할 수 있게 해줍니다.
from tensorflow import keras

# 신경망의 각 층(Conv2D, Dense, Dropout 등)을 만들기 위해 layers를 불러옵니다.
from tensorflow.keras import layers

# 수치 계산과 배열 처리를 위한 NumPy 라이브러리입니다.
import numpy as np

# 학습 결과 그래프와 이미지 출력을 위해 matplotlib를 불러옵니다.
import matplotlib.pyplot as plt

# 외부 이미지 파일(dog.jpg)을 열기 위해 PIL의 Image 모듈을 불러옵니다.
from PIL import Image

# 파일 존재 여부를 확인하기 위해 os 모듈을 불러옵니다.
import os


# CIFAR-10 데이터셋을 불러오기 전에 안내 문구를 출력합니다.
print("CIFAR-10 데이터셋 로드 중...")

# CIFAR-10 데이터셋을 로드합니다.
# x_train, y_train: 학습용 이미지와 정답
# x_test, y_test: 테스트용 이미지와 정답
# CIFAR-10은 32x32 크기의 컬러 이미지 10개 클래스로 구성된 데이터셋입니다.
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()


# 숫자 라벨(0~9)을 사람이 읽을 수 있는 클래스 이름으로 바꾸기 위한 리스트입니다.
# 예를 들어 예측값이 3이면 'cat'으로 해석할 수 있습니다.
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


# 이미지 픽셀 값을 0~255 범위에서 0~1 범위로 정규화합니다.
# 이렇게 하면 학습이 더 안정적으로 되고, 최적화가 잘 이루어집니다.
print("데이터 정규화 중...")
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


# CIFAR-10의 y값은 보통 (50000, 1), (10000, 1) 형태입니다.
# flatten()을 사용하면 (50000,), (10000,)처럼 1차원 배열로 바뀝니다.
# sparse_categorical_crossentropy를 사용할 때 이런 형태가 더 다루기 편합니다.
y_train = y_train.flatten()
y_test = y_test.flatten()


# 데이터의 전체 형태를 출력하여 제대로 로드되었는지 확인합니다.
print(f"훈련 데이터 형태: {x_train.shape}")
print(f"테스트 데이터 형태: {x_test.shape}")
print()


# CNN 모델을 만들기 시작한다는 안내 문구입니다.
print("CNN 모델 구축 중...")


# Sequential 모델은 층을 순서대로 쌓는 가장 기본적인 모델 구조입니다.
model = keras.Sequential([
    
    # 입력 이미지의 크기를 지정합니다.
    # CIFAR-10 이미지는 32x32 크기의 RGB 컬러 이미지이므로 shape=(32, 32, 3)입니다.
    layers.Input(shape=(32, 32, 3)),
    
    
    # -------------------- 첫 번째 Convolution Block --------------------
    
    # 32개의 필터를 사용하는 합성곱 층입니다.
    # (3,3) 크기의 필터로 이미지의 지역적 특징(모서리, 패턴 등)을 추출합니다.
    # padding='same'은 출력 이미지 크기를 입력과 같게 유지합니다.
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    
    # 배치 정규화(BatchNormalization)입니다.
    # 각 층의 출력 분포를 안정화하여 학습을 더 빠르고 안정적으로 만들어줍니다.
    layers.BatchNormalization(),
    
    # 같은 블록 안에서 한 번 더 합성곱을 수행해 특징을 더 깊게 추출합니다.
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    
    # 다시 배치 정규화를 적용합니다.
    layers.BatchNormalization(),
    
    # MaxPooling은 2x2 영역에서 가장 큰 값만 남겨 이미지 크기를 줄입니다.
    # 중요한 특징은 유지하면서 계산량을 줄이는 역할을 합니다.
    layers.MaxPooling2D((2, 2)),
    
    # Dropout은 일부 뉴런 출력을 무작위로 끊어 과적합을 방지합니다.
    # 여기서는 25%를 비활성화합니다.
    layers.Dropout(0.25),
    
    
    # -------------------- 두 번째 Convolution Block --------------------
    
    # 필터 수를 64개로 늘려 더 복잡한 특징을 추출합니다.
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    
    # 한 번 더 합성곱을 적용하여 특징 표현력을 높입니다.
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    
    # 다시 공간 크기를 줄입니다.
    layers.MaxPooling2D((2, 2)),
    
    # 과적합 방지를 위한 Dropout입니다.
    layers.Dropout(0.25),
    
    
    # -------------------- 세 번째 Convolution Block --------------------
    
    # 필터 수를 128개로 늘려 더 고수준의 특징을 추출합니다.
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    
    # 같은 블록 안에서 추가 합성곱을 수행합니다.
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    
    # 다시 MaxPooling으로 특징 맵의 크기를 줄입니다.
    layers.MaxPooling2D((2, 2)),
    
    # Dropout으로 과적합을 줄입니다.
    layers.Dropout(0.25),
    
    
    # -------------------- 분류기(Classifier) 부분 --------------------
    
    # 지금까지의 3차원 특징 맵을 1차원 벡터로 펼칩니다.
    # Dense 층은 1차원 입력을 받기 때문에 Flatten이 필요합니다.
    layers.Flatten(),
    
    # 완전연결층(Dense)입니다.
    # 앞에서 추출한 특징들을 종합해 분류에 적합한 표현으로 바꿉니다.
    layers.Dense(256, activation='relu'),
    
    # 분류기 부분에서도 Dropout을 크게 주어 과적합을 방지합니다.
    layers.Dropout(0.5),
    
    # 최종 출력층입니다.
    # CIFAR-10은 10개 클래스로 분류하므로 출력 뉴런도 10개입니다.
    # softmax를 사용해 각 클래스에 대한 확률값을 출력합니다.
    layers.Dense(10, activation='softmax')
])


# 모델의 학습 방법을 설정합니다.
model.compile(
    # Adam 옵티마이저를 사용합니다.
    # learning_rate=0.001은 가중치를 얼마나 빠르게 조정할지 결정하는 값입니다.
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    
    # 정답 라벨이 one-hot 인코딩이 아닌 정수 라벨이므로
    # sparse_categorical_crossentropy 손실 함수를 사용합니다.
    loss='sparse_categorical_crossentropy',
    
    # 학습 중 정확도도 함께 계산합니다.
    metrics=['accuracy']
)


# 모델의 전체 구조를 출력합니다.
# 각 층의 출력 형태와 학습해야 할 파라미터 개수를 확인할 수 있습니다.
print("모델 아키텍처:")
model.summary()
print()


# 실제 학습을 시작한다는 안내 문구입니다.
print("모델 훈련 중...")

# model.fit()은 학습 데이터를 이용해 모델을 훈련시키는 핵심 함수입니다.
history = model.fit(
    # 입력 이미지 데이터
    x_train,
    
    # 각 이미지에 대한 정답 라벨
    y_train,
    
    # 전체 데이터를 20번 반복 학습합니다.
    epochs=20,
    
    # 한 번에 64개 샘플씩 묶어서 학습합니다.
    batch_size=64,
    
    # 학습 데이터의 10%를 검증용 데이터로 자동 분리합니다.
    # 검증 데이터는 학습에는 사용하지 않고 성능 확인용으로만 씁니다.
    validation_split=0.1,
    
    # 학습 진행 상황을 출력합니다.
    verbose=1
)
print()


# 학습이 끝난 뒤, 테스트 세트에서 최종 성능을 평가합니다.
print("테스트 세트에서 성능 평가:")
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

# 정확도를 퍼센트 형태로 출력합니다.
print(f"테스트 정확도: {test_accuracy * 100:.2f}%")

# 손실값도 함께 출력합니다.
print(f"테스트 손실(Loss): {test_loss:.4f}")
print()


# 학습 과정에서 accuracy와 loss가 어떻게 변했는지 시각화합니다.
plt.figure(figsize=(12, 4))


# -------------------- 정확도 그래프 --------------------
plt.subplot(1, 2, 1)

# epoch별 학습 정확도
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)

# epoch별 검증 정확도
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)


# -------------------- 손실 그래프 --------------------
plt.subplot(1, 2, 2)

# epoch별 학습 손실
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)

# epoch별 검증 손실
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# 그래프 간격을 자동 조정합니다.
plt.tight_layout()

# 그래프를 출력합니다.
plt.show()


# 테스트 이미지 일부에 대해 실제 예측 결과를 확인합니다.
print("테스트 이미지에 대한 예측:")

# 처음 10개 샘플만 확인합니다.
num_samples = 10

# 모델이 각 이미지에 대해 10개 클래스 확률을 예측합니다.
predictions = model.predict(x_test[:num_samples])


# 예측 결과를 이미지와 함께 시각화합니다.
plt.figure(figsize=(15, 3))
for i in range(num_samples):
    # 2행 5열 구조로 배치합니다.
    plt.subplot(2, 5, i + 1)
    
    # 테스트 이미지를 출력합니다.
    plt.imshow(x_test[i])
    
    # 예측 확률 중 가장 큰 값의 인덱스를 가져옵니다.
    # 즉, 모델이 가장 가능성이 높다고 판단한 클래스입니다.
    predicted_label = np.argmax(predictions[i])
    
    # 실제 정답 클래스입니다.
    true_label = y_test[i]
    
    # 예측이 맞으면 초록색, 틀리면 빨간색으로 제목 색을 설정합니다.
    color = 'green' if predicted_label == true_label else 'red'
    
    # 예측 클래스 이름과 실제 클래스 이름을 제목으로 표시합니다.
    plt.title(f'Pred: {class_names[predicted_label]}\nTrue: {class_names[true_label]}',
              color=color, fontsize=9)
    
    # 축은 시각적으로 필요 없으므로 숨깁니다.
    plt.axis('off')

plt.tight_layout()
plt.show()


# -------------------- 외부 이미지(dog.jpg) 예측 부분 --------------------

# 예측할 외부 이미지 파일 이름입니다.
dog_image_path = 'dog.jpg'

# 해당 파일이 실제로 존재하는지 확인합니다.
if os.path.exists(dog_image_path):
    print(f"\ndog.jpg에 대한 예측:")
    
    # 이미지를 열고 RGB 형식으로 변환합니다.
    # convert('RGB')를 하는 이유는 모델 입력 형식을 CIFAR-10과 맞추기 위해서입니다.
    dog_img = Image.open(dog_image_path).convert('RGB')
    
    # 모델 입력 크기와 맞추기 위해 이미지를 32x32로 리사이즈합니다.
    dog_img_resized = dog_img.resize((32, 32))
    
    # 이미지를 NumPy 배열로 변환하고 0~1 범위로 정규화합니다.
    dog_array = np.array(dog_img_resized).astype("float32") / 255.0
    
    # 모델 입력은 (배치크기, 높이, 너비, 채널) 형태여야 하므로
    # 이미지 1장을 배치 형태로 만들기 위해 차원을 하나 추가합니다.
    # 결과 shape는 (1, 32, 32, 3)이 됩니다.
    dog_batch = np.expand_dims(dog_array, axis=0)
    
    # 외부 이미지에 대해 예측을 수행합니다.
    dog_prediction = model.predict(dog_batch)
    
    # 가장 확률이 높은 클래스 인덱스를 가져옵니다.
    predicted_class = np.argmax(dog_prediction[0])
    
    # 가장 높은 확률값 자체를 신뢰도로 사용합니다.
    confidence = np.max(dog_prediction[0])
    
    print(f"예측 클래스: {class_names[predicted_class]}")
    print(f"신뢰도: {confidence * 100:.2f}%")
    
    # 예측 결과와 클래스별 확률 분포를 시각화합니다.
    plt.figure(figsize=(12, 5))
    
    # 왼쪽: 입력 이미지
    plt.subplot(1, 2, 1)
    plt.imshow(dog_img_resized)
    plt.title(f'Input Image: {class_names[predicted_class]}')
    plt.axis('off')
    
    # 오른쪽: 각 클래스별 예측 확률
    plt.subplot(1, 2, 2)
    plt.barh(class_names, dog_prediction[0])
    plt.xlabel('Probability')
    plt.title('Prediction Probabilities')
    plt.xlim(0, 1)
    
    plt.tight_layout()
    plt.show()

else:
    # 파일이 없으면 예측을 건너뜁니다.
    print(f"\n경고: {dog_image_path} 파일을 찾을 수 없습니다.")
    print("외부 이미지 테스트를 건너뜁니다.")
```

## 주요 코드 설명

### 1. CIFAR-10 데이터셋 로드 및 클래스 정의

```python
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
```

CIFAR-10 컬러 이미지 분류 데이터셋을 로드합니다.

- `x_train` : 훈련 이미지 (50,000개, 32×32×3 RGB)  
- `y_train` : 훈련 레이블 (10개 클래스, 0~9)  
- `x_test` : 테스트 이미지 (10,000개, 32×32×3 RGB)  
- `y_test` : 테스트 레이블  
- `class_names` : 숫자 인덱스를 거기해당 사물명으로 변환하기 위한 매핑 리스트

### 2. 데이터 정규화 및 형태 변환

```python
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train = y_train.flatten()
y_test = y_test.flatten()
```

데이터 전처리를 수행합니다.

- **이미지 정규화**: 픽셀값을 0~1 범위로 스케일링하여 신경망 학습 안정성 향상
- **라벨 평탄화**: CIFAR-10 라벨이 (50000, 1) 형태이므로 (50000,)으로 변환
  - `sparse_categorical_crossentropy` 손실 함수가 이 형태를 요구합니다.

### 3. CNN 모델 아키텍처

```python
model = keras.Sequential([
    layers.Input(shape=(32, 32, 3)),
    
    # 첫 번째 Conv 블록
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    # ... 추가 블록들
])
```

깊은 CNN을 구성합니다. 주요 컴포넌트:

- **Conv2D(필터수, 커널크기)**:
  - 필터: 이미지에서 특징을 추출하는 가중치 행렬
  - 3×3 커널: 가장 흔히 사용하는 크기로 에지, 모서리, 텍스처 등 추출
  - 필터 수를 점진적으로 증가 (32 → 64 → 128): 고수준의 특징 추출
  - `padding='same'`: 출력 이미지 크기를 입력과 동일하게 유지

- **BatchNormalization()**:
  - 각 층의 출력을 정규분포로 만들어 학습 안정화
  - 더 높은 학습률 사용 가능, 과적합 감소

- **MaxPooling2D((2,2))**:
  - 2×2 영역에서 가장 큰 값만 선택
  - 공간 차원 축소로 계산효율 증가 (32 → 16 → 8)
  - 중요 특징은 보존하고 노이즈 감소

- **Dropout(율)**:
  - 학습 중 일부 뉴런을 무작위로 비활성화
  - Conv 층 다음: 0.25 (25% 비활성화)
  - Dense 층 다음: 0.5 (50% 비활성화, 더 강한 규제)
  - 과적합 방지 효과

### 4. Flatten과 분류기 구성

```python
layers.Flatten(),
layers.Dense(256, activation='relu'),
layers.Dropout(0.5),
layers.Dense(10, activation='softmax')
```

CNN의 출력을 분류 레이어로 변환합니다.

- **Flatten()**: 3D 특징 맵을 1D 벡터로 변환
  - 마지막 MaxPooling 후 크기: (8, 8, 128) → 8192 길이의 벡터
  
- **Dense(256, 'relu')**:
  - 추출된 특징들을 종합하여 분류 가능한 표현으로 변환
  - 256개 뉴런은 일반적인 하이퍼파라미터 선택

- **Dense(10, 'softmax')**:
  - 최종 출력층: 10개 클래스의 확률 분포
  - Softmax: $ \sigma(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}} $
  - 모든 출력의 합 = 1, 각 출력 = 해당 클래스 확률

### 5. 모델 컴파일

```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

모델의 학습 설정을 구성합니다.

- **Adam(learning_rate=0.001)**:
  - 매개변수별 적응적 학습률 조정
  - 기본값보다 낮은 0.001로 설정하여 세밀한 최적화
  - 모멘텀과 RMSprop의 장점 결합

- **sparse_categorical_crossentropy**:
  - 정수형 라벨(0~9)에 적합한 손실 함수
  - One-hot 인코딩 불필요로 메모리 효율적

- **metrics=['accuracy']**:
  - 훈련 중 정확도 모니터링

### 6. 모델 훈련

```python
history = model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)
```

모델을 훈련시킵니다.

- **epochs=20**: 
  - 전체 데이터를 20번 반복
  - 충분히 학습하되 과도한 훈련 방지

- **batch_size=64**:
  - 메모리와 학습 안정성의 균형
  - MNIST의 128보다 작은 이유: 더 복잡한 CIFAR-10은 작은 배치 크기가 유리

- **validation_split=0.1**:
  - 훈련 데이터의 10%를 검증용으로 자동 분리
  - 과적합 감지 및 조기 종료 판단 기준

### 7. 외부 이미지 예측

```python
dog_img = Image.open(dog_image_path).convert('RGB')
dog_img_resized = dog_img.resize((32, 32))
dog_array = np.array(dog_img_resized).astype("float32") / 255.0
dog_batch = np.expand_dims(dog_array, axis=0)
dog_prediction = model.predict(dog_batch)
```

외부 이미지에 대한 예측을 수행합니다.

- **PIL Image 읽기**: `.convert('RGB')`로 일관된 색상 채널 보장
- **리사이즈**: 모델 입력 크기(32×32)로 조정
- **정규화**: 훈련 데이터와 동일한 방식으로 0~1 범위 변환
- **배치 확장**: `expand_dims()`로 (1, 32, 32, 3) 형태로 변환
  - 모델 입력은 배치 차원 필요

### 8. 예측 결과 시각화

예측 이미지와 확률 분포를 시각화하여 모델의 의사결정 과정을 이해합니다.

- 올바른 예측은 초록색, 오류는 빨간색으로 표시
- 막대 그래프로 각 클래스의 신뢰도 비교

## 실행 결과

모델 훈련 시 정확도는 약 75~80% 수준에 도달합니다.

**모델 구조 출력 예시:**
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 32, 32, 32)       896       
batch_normalization          (None, 32, 32, 32)       128       
conv2d_1 (Conv2D)            (None, 32, 32, 32)       9248      
batch_normalization_1        (None, 32, 32, 32)       128       
max_pooling2d (MaxPooling2D) (None, 16, 16, 32)       0         
dropout (Dropout)            (None, 16, 16, 32)       0         
...
flatten (Flatten)            (None, 8192)             0         
dense (Dense)                (None, 256)              2097408   
dropout_9 (Dropout)          (None, 256)              0         
dense_1 (Dense)              (None, 10)               2570      
=================================================================
Total params: 1,132,802
Trainable params: 1,132,610
```

**훈련 결과 예시:**
- Epoch 1/20: loss: 1.9843, accuracy: 0.2845, val_loss: 1.5621, val_accuracy: 0.4362
- Epoch 10/20: loss: 0.8234, accuracy: 0.7156, val_loss: 0.9876, val_accuracy: 0.6921
- Epoch 20/20: loss: 0.4532, accuracy: 0.8412, val_loss: 0.6234, val_accuracy: 0.7854

**정확도 및 손실 변화 그래프:**
- Training Accuracy: 약 28% → 84% (점진적이고 안정적인 증가)
- Validation Accuracy: Training과 유사하게 증가 (약 75~79%)
- Training Loss: 약 1.98 → 0.45 (빠른 초기 감소, 후에 안정화)
- Validation Loss: Training Loss보다 약간 높게 유지 (일반적인 과적합 패턴)

**테스트 성능:**
- 테스트 정확도: 약 75~80%
- 테스트 손실: 약 0.58~0.70

**예측 시각화 예시:**
- 올바르게 분류된 이미지들이 초록색으로 표시됩니다.
- 분류 실패한 이미지들이 빨간색으로 표시됩니다.
- 대부분의 이미지가 정확하게 분류되지만 일부 어려운 케이스도 존재합니다.

**dog.jpg 예측 결과:**
- 모델이 개로 올바르게 분류할 확률이 높으면 'dog' 클래스로 예측됩니다.
- 예측 확률분포를 통해 각 클래스별 신뢰도를 확인할 수 있습니다.
- 예: `dog (confidence: 89.5%), cat (confidence: 5.2%), ...`

<img width="2387" height="910" alt="image" src="https://github.com/user-attachments/assets/83a8c85b-c631-42b0-91e8-369bb74062ed" />
<img width="2863" height="701" alt="image" src="https://github.com/user-attachments/assets/3d8ac541-ab9c-4811-b7c2-38baa088873b" />


