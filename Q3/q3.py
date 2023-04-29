
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
train_dir = Path('./charts/train_val')
val_dir = Path('./charts/train_val')
# Load the train and validation images and labels
X_train, y_train = getImages(train_dir)
X_val, y_val = getImages(val_dir)


#creating model
model = keras.Sequential([
    
    layers.Conv2D(30, (3,3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2,2)),
 
    layers.Flatten(),
    layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))

print('Test accuracy:', test_acc)
print('Test loss:', test_loss)



