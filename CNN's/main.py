# Импортиране на библиотеки
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Зареждане на CIFAR-10 данни
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Нормализиране на данните
x_train, x_test = x_train / 255.0, x_test / 255.0

# Преобразуване на етикетите към one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Създаване на модела
model = models.Sequential()

# Първи свиващ слой
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))

# Втори свиващ слой
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Трети свиващ слой
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Flatten слой
model.add(layers.Flatten())

# Напълно свързан слой
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Компилиране на модела
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Обучение на модела
history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

# Оценка на модела
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Точност на тестовете: {test_acc:.2f}")

# Визуализация на резултатите
plt.plot(history.history['accuracy'], label='Точност при обучение')
plt.plot(history.history['val_accuracy'], label='Точност при валидиране')
plt.xlabel('Епохи')
plt.ylabel('Точност')
plt.legend()
plt.show()

# Прогноза върху тестови данни
predictions = model.predict(x_test[:5])

# Показване на резултати
for i in range(5):
    plt.imshow(x_test[i])
    plt.title(f"Предсказано: {predictions[i].argmax()}, Истинско: {y_test[i].argmax()}")
    plt.show()
