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
x_train = x_train / 255.0
x_test = x_test / 255.0

# 3. CNN 모델 구성
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
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
    img = img.resize((32, 32))  # CIFAR-10 크기에 맞춤
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction[0])
    confidence = np.max(prediction[0])

    print(f"예측 결과: {class_names[predicted_class]}")
    print(f"신뢰도: {confidence:.4f}")

# 예시 실행
predict_image("dog.jpg")