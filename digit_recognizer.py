import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0 
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)


# buid cnn
model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    keras.layers.MaxPooling2D(2,2),

    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),

    keras.layers.Flatten(),

    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# train
print("Training...")
model.fit(x_train, y_train, epochs=3)


# test acc
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Accuracy:", test_acc)


# load image
def load_image(path):
    img = Image.open(path).convert('L') 
    img = img.resize((28, 28))

    img = np.array(img)

    if img.mean() > 127:
        img = 255 - img

    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)

    return img

# predick
img = load_image("test.png")

prediction = model.predict(img)
digit = np.argmax(prediction)

print("Bố mày dự đoán là số:", digit)

# show cai phan tich
plt.imshow(img.reshape(28,28), cmap='gray')
plt.title(f"Predict: {digit}")
plt.show()