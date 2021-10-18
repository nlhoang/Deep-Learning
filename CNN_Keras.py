import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

# Add dataset MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_val, y_val = X_train[50000:60000, :], y_train[50000:60000]
X_train, y_train = X_train[:50000, :], y_train[:50000]

# Reshape data
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_val = X_val.reshape(X_val.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

Y_train = np_utils.to_categorical(y_train, 10)
Y_val = np_utils.to_categorical(y_val, 10)
Y_test = np_utils.to_categorical(y_test, 10)

# Define model
model = Sequential()
# Add Conv-layer: 32 kernel size 3*3, sigmoid and input shape
model.add(Conv2D(32, (3, 3), activation='sigmoid', input_shape=(28, 28, 1)))
model.add(Conv2D(32, (3, 3), activation='sigmoid'))
# Add Max Pooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))
# Add Flatten tensor -> vector
model.add(Flatten())
# Add Fully Connected layer with 128 nodes and sigmoid
model.add(Dense(128, activation='sigmoid'))
# Output with softmax
model.add(Dense(10, activation='softmax'))

# Compile with loss function
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
H = model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1, validation_data=(X_val, Y_val))

# Plt Loss, accuracy
fig = plt.figure()
numOfEpoch = 10
plt.plot(np.arange(0, numOfEpoch), H.history['loss'], label='training loss')
plt.plot(np.arange(0, numOfEpoch), H.history['val_loss'], label='validation loss')
plt.plot(np.arange(0, numOfEpoch), H.history['accuracy'], label='accuracy')
plt.plot(np.arange(0, numOfEpoch), H.history['val_accuracy'], label='validation accuracy')
plt.title('Accuracy and Loss')
plt.xlabel('Epoch')
plt.ylabel('LossAccuracy')
plt.legend()

score = model.evaluate(X_test, Y_test, verbose=0)
print(score)

plt.imshow(X_test[0].reshape(28, 28), cmap='gray')
y_predict = model.predict(X_test[0].reshape(1, 28, 28, 1))
print(np.argmax(y_predict))
"""
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(X_train[i+9])
plt.show()
"""
