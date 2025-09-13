import datetime
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras

import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('MacOSX')

print(tf.__version__)
print(keras.__version__)

# Data Load
mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Modeling
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training
model.fit(x_train, y_train, epochs=10)
model.evaluate(x_test, y_test)

# Visualize
idx = np.random.randint(len(x_train))
image = x_train[idx]

plt.imshow(image, cmap='gray')
plt.title(y_train[idx])
plt.show()

#============================================================

# 1. Tensor
# Multi-dimensional array 를 나타내는 말, TensorFlow의 기본 데이터 타입
hello = tf.constant([3, 3], dtype=tf.float32)
print(hello)

x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
print(x)
print(type(x))

x_np = np.array([[1.0, 2.0],
                 [3.0, 4.0]])

x_list = [[1.0, 2.0],
          [3.0, 4.0]]

x.numpy()
print(type(x.numpy()))
#============================================================

# 2. Dataset
fashion_mnist = keras.datasets.fashion_mnist
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]

(train_image, train_label), (test_image, test_label) = fashion_mnist.load_data()

# shape 확인
print(train_image.shape, train_label.shape)
print(test_image.shape, test_label.shape)

unique, counts = np.unique(train_label, axis=-1, return_counts=True)
print(dict(zip(unique, counts)))

unique, counts = np.unique(test_label, axis=-1, return_counts=True)
print(dict(zip(unique, counts)))

# 학습데이터 시각화
plt.figure(figsize=(8,8))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_image[i], cmap='gray')
    plt.title(class_names[train_label[i]])
plt.show()

# 데이터 전처리
## 각 이미지를 0~1사이 값으로 만들어주기 위해 /255.0 을 수행
train_image = train_image.astype(np.float32) / 255.0
test_image = test_image.astype(np.float32) / 255.0

## One hot Encoding
train_label = keras.utils.to_categorical(train_label)
test_label = keras.utils.to_categorical(test_label)

# Dataset 생성
train_dataset = tf.data.Dataset.from_tensor_slices((train_image, train_label)).shuffle(buffer_size=100000).batch(64)
test_dataset = tf.data.Dataset.from_tensor_slices((test_image, test_label)).batch(64)

## Dataset Iteration
imgs, lbs = next(iter(train_dataset))
print("Feature Batch shape:", imgs.shape)
print("Label Batch shape:", lbs.shape)

img = imgs[0]
lb = lbs[0]

plt.imshow(img, cmap='gray')
plt.show()
print(f"Label: {lb}")


# Custom dataset
a = np.arange(10)
print(a)

ds_tensor = tf.data.Dataset.from_tensor_slices(a)
print(ds_tensor)

for x in ds_tensor:
    print(x)

# 데이터 전처리
ds_tensors = ds_tensor.map(tf.square).shuffle(10).batch(2)

for _ in range(3):
    for x in ds_tensors:
        print(x)
    print("=" * 50)

#============================================================

# 3. Modeling
## 3-1. Keras Sequential API
def create_seq_model():
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model

seq_model = create_seq_model()
seq_model.summary()

## 3-2. Keras Functional API
def create_func_model():
    # 각 Layer 정의
    inputs = keras.Input(shape=(28, 28))
    flatten = keras.layers.Flatten()(inputs)
    dense = keras.layers.Dense(128, activation='relu')(flatten)
    dropout = keras.layers.Dropout(0.2)(dense)
    outputs = keras.layers.Dense(10, activation='softmax')(dropout)

    # 모델의 시작과 끝을 정의
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

func_model = create_func_model()
func_model.summary()

## 3-3. Model Class: Subclassing
class SubClassModel(keras.Model):
    def __init__(self):
        super(SubClassModel, self).__init__()

        self.flatten = keras.layers.Flatten(input_shape=(28, 28))
        self.dense1 = keras.layers.Dense(128, activation='relu')
        self.dropout = keras.layers.Dropout(0.2)
        self.dense2 = keras.layers.Dense(10, activation='softmax')

    def call(self, x, training=False):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)

        return self.dense2(x)

subclass_model = SubClassModel()

inputs = tf.zeros((1, 28, 28))
subclass_model(inputs)

subclass_model.summary()

inputs = tf.random.normal((1, 28, 28))
outputs = subclass_model(inputs)
pred = tf.argmax(outputs, -1)
print(f"Predicted Class: {pred}")

#============================================================

# 4. Training / Validation
## 4-1. Keras API
learning_rate = 0.001

seq_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

history = seq_model.fit(train_dataset, epochs=10, validation_data=test_dataset)

### Plot Loss
plt.plot(history.history['loss'], 'b-', label='Loss')
plt.plot(history.history['val_loss'], 'r--', label='Validation Loss')
plt.xlabel('Epochs')
plt.legend()
plt.show()

### Plot Accuracy
plt.plot(history.history['accuracy'], 'b-', label='Accuracy')
plt.plot(history.history['val_accuracy'], 'r--', label='Validation Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.show()


## 4-2. GradientTape
# Loss Func. 정의
loss_object = keras.losses.CategoricalCrossentropy(from_logits=True)

# optimizer 정의
learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

train_loss = keras.metrics.Mean(name="train_loss")
train_accuracy = keras.metrics.CategoricalAccuracy(name="train_accuracy")

test_loss = keras.metrics.Mean(name="test_loss")
test_accuracy = keras.metrics.CategoricalAccuracy(name="test_accuracy")

@tf.function
def train_step(model, images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)   # Back Propagation
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(model, images, labels):
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

# training
EPOCHS = 10

for epoch in range(EPOCHS):
    train_loss.reset_state()
    train_accuracy.reset_state()

    test_loss.reset_state()
    test_accuracy.reset_state()

    for images, labels in train_dataset:
        train_step(func_model, images, labels)

    for test_images, test_labels in test_dataset:
        test_step(func_model, test_images, test_labels)

    print(
        f'Epochs {epoch + 1}, '
        f'Loss: {train_loss.result()}, '
        f'Accuracy: {train_accuracy.result() * 100}, '
        f'Test Loss: {test_loss.result()}, '
        f'Test Accuracy: {test_accuracy.result() * 100}'
    )

#============================================================

# 5. Model 저장/불러오기
# 5-1. 가중치 저장/불러오기
seq_model.save_weights("DeepLearning/Ver2022/3_Tensorflow_Torch_Basic/seq_model.weights.h5")

seq_model2 = create_seq_model()
seq_model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])
seq_model2.evaluate(test_dataset)

seq_model2.load_weights("DeepLearning/Ver2022/3_Tensorflow_Torch_Basic/seq_model.weights.h5")
seq_model2.evaluate(test_dataset)

## 5-2. 모델 전체 저장/불러오기
seq_model.save("DeepLearning/Ver2022/3_Tensorflow_Torch_Basic/seq_model.keras")

seq_model3 = keras.models.load_model("DeepLearning/Ver2022/3_Tensorflow_Torch_Basic/seq_model.keras")
seq_model3.evaluate(test_dataset)

## 5-3. Tensorboard 사용하여 시각화하기
new_model1 = create_seq_model()
new_model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])
new_model1.evaluate(test_dataset)

log_dir = "DeepLearning/Ver2022/3_Tensorflow_Torch_Basic/logs/new_model1"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
new_model1.fit(train_dataset, epochs=EPOCHS, validation_data=test_dataset, callbacks=[tensorboard_callback])
# terminal command: tensorboard --logdir=DeepLearning/Ver2022/3_Tensorflow_Torch_Basic/logs/new_model1

## 5-4. Summary Writer 사용하기
new_model2 = create_seq_model()

loss_object = keras.losses.CategoricalCrossentropy()

learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

train_loss = keras.metrics.Mean(name="train_loss")
train_accuracy = keras.metrics.CategoricalAccuracy(name="train_accuracy")

test_loss = keras.metrics.Mean(name="test_loss")
test_accuracy = keras.metrics.CategoricalAccuracy(name="test_accuracy")


@tf.function
def train_step(model, images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(model, images, labels):
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = "DeepLearning/Ver2022/3_Tensorflow_Torch_Basic/logs/gradient_tape/" + current_time + "/train"
test_log_dir = "DeepLearning/Ver2022/3_Tensorflow_Torch_Basic/logs/gradient_tape/" + current_time + "/test"

train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

Epochs = 10

for epoch in range(Epochs):
    train_loss.reset_state()
    train_accuracy.reset_state()
    test_loss.reset_state()
    test_accuracy.reset_state()

    for images, labels in train_dataset:
        train_step(func_model, images, labels)
    with train_summary_writer.as_default():
        tf.summary.scalar("loss", train_loss.result(), step=epoch)
        tf.summary.scalar("accuracy", train_accuracy.result(), step=epoch)

    for test_images, test_labels in test_dataset:
        test_step(func_model, test_images, test_labels)
    with test_summary_writer.as_default():
        tf.summary.scalar("loss", test_loss.result(), step=epoch)
        tf.summary.scalar("accuracy", test_accuracy.result(), step=epoch)

    print(
        f"Epochs {epoch + 1}, "
        f"Loss: {train_loss.result()}, "
        f"Accuracy: {train_accuracy.result() * 100}, "
        f"Test Loss: {test_loss.result()}, "
        f"Test Accuracy: {test_accuracy.result() * 100}"
    )
# terminal command: tensorboard --logdir=./logs/gradient_tape/{current_time}
