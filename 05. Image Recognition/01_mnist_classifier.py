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

