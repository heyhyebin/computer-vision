# 5 주차
## 01. 간단한 이미지 분류기 구현
### 문제
#### 설명
• 손글씨 숫자 이미지(MNIST 데이터셋)를 이용하여 간단한 이미지 분류기를 구현             

#### 요구사항
• MNIST 데이터셋을 로드             
• 데이터를 훈련 세트와 테스트 세트로 분할             
• 간단한 신경망 모델을 구축             
• 모델을 훈련시키고 정확도를 평가             

#### 힌트
• tensorflow.keras.datasets에서 MNIST 데이터셋을 불러올 수 있음             
• Sequential 모델과 Dense 레이어를 활용하여 신경망을 구성             
• 손글씨 숫자 이미지는 28x28 픽셀 크기의 흑백 이미지             

### 코드
01_mnist_classifier.py
```python
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. MNIST 데이터셋 불러오기
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. 데이터 전처리
# 픽셀 값을 0~255 -> 0~1 범위로 정규화
x_train = x_train / 255.0
x_test = x_test / 255.0

# 3. 간단한 신경망 모델 구성
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),   # 28x28 이미지를 1차원으로 펼침
    layers.Dense(128, activation='relu'),   # 은닉층
    layers.Dense(10, activation='softmax')  # 출력층 (0~9 숫자 10개)
])

# 4. 모델 컴파일
model.compile(
    optimizer='adam',       # 최적화 알고리즘
    loss='sparse_categorical_crossentropy', # 손실 함수
    metrics=['accuracy']                    # 평가 지표
)

# 5. 모델 학습
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# 6. 모델 평가
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

print(f"\n테스트 정확도: {test_acc:.4f}")
```

### 핵심코드       
    • layers.Flatten(input_shape=(28, 28))
      2차원 이미지(28x28)를 1차원 벡터로 변환하는 코드

    • layers.Dense(128, activation='relu')
      이미지 특징을 학습하는 은닉층
    
    • layers.Dense(10, activation='softmax')
      0~9 숫자를 분류하는 출력층

    • model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)
      모델을 학습시키는 핵심 코드
    
    • model.evaluate(x_test, y_test)
      테스트 데이터로 모델 성능을 평가하는 코드



### 실행결과
<img width="853" height="222" alt="1" src="https://github.com/user-attachments/assets/31aa0bcb-ab6c-43b3-a2c7-f22147f4ed61" />












## 02. CIFAR-10 데이터셋을활용한CNN 모델구축
### 문제
#### 설명
* CIFAR-10 데이터셋을 활용하여 합성곱 신경망(CNN)을 구축하고, 이미지 분류를 수행             

#### 요구사항
• CIFAR-10 데이터셋을 로드             
• 데이터 전처리(정규화 등)를 수행             
• CNN 모델을 설계하고 훈련             
• 모델의 성능을 평가하고, 테스트 이미지(dog.jpg)에 대한 예측을 수행             

#### 힌트
• tensorflow.keras.datasets에서 CIFAR-10 데이터셋을 불러올 수 있음             
• Conv2D, MaxPooling2D, Flatten, Dense 레이어를 활용하여 CNN을 구성             
• 데이터 전처리 시 픽셀 값을 0~1 범위로 정규화하면 모델의 수렴이 빨라질 수 있음             

### 코드
02_cifar10_cnn.py
```python
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image

# CIFAR-10 클래스 이름
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# 1. CIFAR-10 데이터셋 불러오기
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# 2. 데이터 전처리
# 픽셀 값을 0~255 -> 0~1 범위로 정규화
x_train = x_train / 255.0
x_test = x_test / 255.0

# 3. CNN 모델 구성
model = models.Sequential([
    # 합성곱 층: 이미지 특징 추출
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),    # 특징 축소

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),

    layers.Flatten(),   # 1차원으로 변환
    layers.Dense(64, activation='relu'),    # 완전연결층
    layers.Dense(10, activation='softmax')  # 10개 클래스 분류
])

# 4. 모델 컴파일
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 5. 모델 학습
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.1)

# 6. 모델 평가
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\n테스트 정확도: {test_acc:.4f}")

# 7. dog.jpg 이미지 예측 함수
def predict_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((32, 32))          # CIFAR-10 크기에 맞춤
    img_array = np.array(img) / 255.0   # numpy 배열로 변환 후 정규화
    img_array = np.expand_dims(img_array, axis=0)  # 모델 입력 형태로 변환 (배치 차원 추가)

    prediction = model.predict(img_array)

    # 가장 높은 확률을 가진 클래스 선택
    predicted_class = np.argmax(prediction[0])
    confidence = np.max(prediction[0])

    print(f"예측 결과: {class_names[predicted_class]}")
    print(f"신뢰도: {confidence:.4f}")

# 8. dog.jpg 이미지 예측 실행
predict_image("dog.jpg")
```

### 핵심코드   
    • layers.Conv2D(32, (3, 3), activation='relu')
      이미지의 특징을 추출하는 합성곱 계층
    
    • layers.MaxPooling2D((2, 2))
      특징 맵의 크기를 줄여 계산량을 감소시키는 계층
    
    • layers.Flatten()
      2차원 특징 맵을 1차원으로 변환하는 코드
    
    • layers.Dense(64, activation='relu')
      추출된 특징을 학습하는 완전연결층
    
    • layers.Dense(10, activation='softmax')
      10개 클래스 중 하나로 분류하는 출력층
    
    • model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.1)
      CNN 모델을 학습시키는 코드
    
    • np.argmax(prediction[0])
      가장 높은 확률을 가진 클래스를 선택하는 코드

### 실행결과
<img width="844" height="409" alt="2" src="https://github.com/user-attachments/assets/95081560-65c7-4416-aa0b-2f147ea8c11e" />
