
# Import librraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import os
import cv2
from pathlib import Path

#Function to Get Images
def getImages(dir_path):
    # Set the image size
    img_size_val = (32, 32)
    # Create empty lists for the images and labels
    images_val = []
    labels_val = []
    # Loop over each folder from '0' to '9'
    for label in range(10):
        data_dir = os.path.join(dir_path, str(label))
        # Loop over each image in the folder
        for file in os.listdir(data_dir):
            file_dir = os.path.join(data_dir, file)
            if file_dir.endswith(('.tiff','.bmp')):
                # Load the image and resize it to the desired size
                img = cv2.imread(file_dir, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, img_size_val)
                # Append the image and label to the lists
                images_val.append(img)
                labels_val.append(label)
    # Convert the lists to NumPy arrays
    images_val = np.array(images_val)
    labels_val = np.array(labels_val)
    return images_val, labels_val


# Set the path to the folder containing the 'train' and 'val' folders
train_dir = Path('./train')
val_dir = Path('./val')
# Load the train and validation images and labels
X_train, y_train = getImages(train_dir)
X_val, y_val = getImages(val_dir)

# Model Building, compilaing and training:
## First build a model in keras/tensorflow and see what weights and bias values it comes up with. 
## We will than try to reproduce same weights and bias in our plain python implementation of gradient descent. Below is the architecture of our simple neural network
model = keras.Sequential([ keras.layers.Flatten(),keras.layers.Dense(10, input_shape=(1024,),activation = 'sigmoid')])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train,epochs= 100,validation_data=(X_val, y_val))

# scaling and checking the accuracy
x_train_scaled = X_train/255
x_test_scaled = X_val/255
model.fit(x_train_scaled, y_train,epochs= 100, validation_data=(x_test_scaled, y_val))

# evaluate test dataset
model.evaluate(x_test_scaled,y_val)

# predict image
y_predicted = model.predict(x_test_scaled)
plt.matshow(X_val[10])
print('Predicted Value is ',np.argmax(y_predicted[10]))
plt.matshow(X_val[50])
print('Predicted Value is ',np.argmax(y_predicted[50]))
plt.matshow(X_val[70])
print('Predicted Value is ',np.argmax(y_predicted[70]))

# Created model with multiple layes to check the accuracy
# created a multiple dense layers where every input is connected to every other output, the number of inputs are 1024, outputs are 10
# activation function is sigmoid
model2 = keras.Sequential([
    keras.layers.Flatten(input_shape=(32, 32)),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model2.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model2.fit(x_train_scaled, y_train, epochs=25, validation_data=(x_test_scaled, y_val))

# Evaluating the model on the test data
test_loss, test_acc = model2.evaluate(x_test_scaled, y_val)
print('accuracy for test:', test_acc)

# Plot the training and validation accuracy
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot the training and validation loss
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# predicting images
y_predicted = model2.predict(x_test_scaled)
plt.matshow(X_val[140])
print('Predicted Value: ',np.argmax(y_predicted[140]))

plt.matshow(X_val[150])
print('Predicted Value:',np.argmax(y_predicted[150]))

plt.matshow(X_val[170])
print('Predicted Value:',np.argmax(y_predicted[170]))