# SIMPLE IMG CLASSIFICATION MODEL TO BE HOSTED AT GENESISL1 EVM BLOCKCHAIN

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers

# Define input shape
img_size = (64, 64)
input_shape = img_size + (3,)

# Define the model architecture with regularization
model = tf.keras.Sequential([
    layers.Conv2D(8, (3, 3), padding='same', activation='relu', input_shape=input_shape, kernel_regularizer=regularizers.l2(0.001)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(16, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(1, activation='sigmoid')
])

# Add data augmentation to the training data
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_data = train_datagen.flow_from_directory(
    'data',
    target_size=img_size,
    batch_size=32,
    class_mode='binary',
    subset='training',
    seed=123
)

# Define validation data
val_data = tf.keras.preprocessing.image_dataset_from_directory(
    'data',
    validation_split=0.1,
    subset='validation',
    seed=123,
    image_size=img_size,
    batch_size=2
)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with regularization and data augmentation
history = model.fit(train_data, epochs=22, validation_data=val_data)

# Save the model
model.save('image_classifier.h5')
