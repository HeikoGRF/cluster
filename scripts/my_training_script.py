import os
import sys
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from zipfile import ZipFile
import tarfile
import shutil
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout, BatchNormalization
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report
import cv2

# Define dataset paths on Euler
data_csv_path = '/cluster/home/hgraef/deep_learning/images/HAM10000/hmnist_28_28_RGB.csv'
metadata_csv_path = '/cluster/home/hgraef/deep_learning/images/HAM10000/HAM10000_metadata.csv'
image_dir = '/cluster/home/hgraef/deep_learning/images/HAM10000/HAM10000_images_part_2'

# Import Data
data = pd.read_csv(data_csv_path)
y = data['label']
x = data.drop(columns=['label'])

# Exploratory Data Analysis (EDA)
tabular_data = pd.read_csv(metadata_csv_path)

# Frequency Distribution of Classes
sns.countplot(x='dx', data=tabular_data)
plt.xlabel('Disease', size=12)
plt.ylabel('Frequency', size=12)
plt.title('Frequency Distribution of Classes', size=16)
plt.show()

# Oversampling to balance the dataset
oversample = RandomOverSampler()
x, y = oversample.fit_resample(x, y)

# Reshape and normalize the image data
x = np.array(x).reshape(-1, 28, 28, 3)
x = (x - np.mean(x)) / np.std(x)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Define the model with additional layers for regularization
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(28, 28, 3)),
    BatchNormalization(),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(0.3),  # Dropout layer to reduce overfitting

    Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(0.3),  # Dropout layer to reduce overfitting

    Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(0.4),  # Increased dropout for deeper layers

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),  # Final dropout layer
    Dense(7, activation='softmax')  # Output layer with 7 classes
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Set early stopping and checkpointing
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='best_model.keras', monitor='val_accuracy', save_best_only=True)

# Train the model
history = model.fit(X_train, Y_train, validation_split=0.2, batch_size=128, epochs=50, callbacks=[early_stopping, checkpoint])

# Plot Accuracy and Loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Load the best model and evaluate it on the test data
model.load_weights('best_model.keras')
loss, acc = model.evaluate(X_test, Y_test, verbose=2)
print(f'Test accuracy: {acc}')

# Model Inference
classes = {0: 'akiec', 1: 'bcc', 2: 'bkl', 3: 'df', 4: 'nv', 5: 'vasc', 6: 'mel'}

for count, temp in enumerate(os.listdir(image_dir)):
    img_path = os.path.join(image_dir, temp)
    img = cv2.imread(img_path)
    
    if img is not None:
        img_resized = cv2.resize(img, (28, 28))
        img_normalized = img_resized / 255.0
        result = model.predict(img_normalized.reshape(1, 28, 28, 3))
        
        max_prob = max(result[0])
        class_ind = list(result[0]).index(max_prob)
        class_name = classes[class_ind]
        
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f'Predicted Class: {class_name}')
        plt.axis('off')
        plt.show()
        
        if count > 10:
            break
    else:
        print(f'Failed to read image: {img_path}')