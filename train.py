# Training Script for Plant Disease Detection Model

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

# Define paths for the training and validation datasets
dataset_path = 'path/to/dataset'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    dataset_path + '/train',  # This is the target directory
    target_size=(150, 150),   # All images will be resized to 150x150
    batch_size=32,
    class_mode='binary'        # Since we use binary_crossentropy loss, we need binary labels
)

# Create a simple model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, steps_per_epoch=100, epochs=15)
