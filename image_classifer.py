import numpy as np
import random
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
import os
from keras.preprocessing import image

base_path = "C:/Users/HP/Documents/ImageClassifier"

X_train = np.loadtxt(os.path.join(base_path, 'input.csv'), delimiter=',')
Y_train = np.loadtxt(os.path.join(base_path, 'labels.csv'), delimiter=',')

X_test = np.loadtxt(os.path.join(base_path, 'input_test.csv'), delimiter=',')
Y_test = np.loadtxt(os.path.join(base_path, 'labels_test.csv'), delimiter=',')

X_train = X_train.reshape(len(X_train), 100, 100, 3)
Y_train = Y_train.reshape(len(Y_train), 1)

X_test = X_test.reshape(len(X_test), 100, 100, 3)
Y_test = Y_test.reshape(len(Y_test), 1)

X_train = X_train / 255.0
X_test = X_test / 255.0

# print("Shape of X_train:", X_train.shape)
# print("Shape of Y_train:", Y_train.shape)
# print("Shape of X_test:", X_test.shape)
# print("Shape of Y_test:", Y_test.shape)

# idx = random.randint(0, len(X_train) - 1)
# plt.imshow(X_train[idx])
# plt.title("Sample Training Image")
# plt.show()

# CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D((2, 2)),

    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=12, batch_size=64)


# idx2 = random.randint(0, len(Y_test) - 1)
# plt.imshow(X_test[idx2])
# plt.title("Test Image")
# plt.show()

# y_pred = model.predict(X_test[idx2].reshape(1, 100, 100, 3))
# y_pred = y_pred > 0.5
  
# pred = 'cat' if y_pred else 'dog'
# print("Our model says it is a:", pred)


model.save(os.path.join(base_path, 'cat_dog_model1.h5'))
print("Model saved successfully!")



# img_path = os.path.join(base_path, "image.jpg")  # Replace with your own image name

# if os.path.exists(img_path):
#     img = image.load_img(img_path, target_size=(100, 100))
#     plt.imshow(img)
#     plt.title("Your Uploaded Image")
#     plt.axis("off")
#     plt.show()

#     img_array = image.img_to_array(img)
#     img_array = img_array / 255.0
#     img_array = np.expand_dims(img_array, axis=0)  # Shape becomes (1, 100, 100, 3)

#     y_pred_custom = model.predict(img_array)
#     pred_custom = 'cat' if y_pred_custom > 0.5 else 'dog'
#     print("Prediction on your image:", pred_custom)
# else:
#     print("Custom image not found at:", img_path)