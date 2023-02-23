import tensorflow as tf
from tensorflow.keras import layers

# Define input shape
img_size = (128, 128)
input_shape = img_size + (3,)

# Define the model architecture
model = tf.keras.Sequential([
    layers.Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    'data',
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=img_size,
    batch_size=2
)

val_data = tf.keras.preprocessing.image_dataset_from_directory(
    'data',
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=img_size,
    batch_size=2
)

history = model.fit(train_data, epochs=69, validation_data=val_data)

# Apply post-training quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the quantized model
with open('model_quantized.tflite', 'wb') as f:
    f.write(tflite_model)

# Save the model regular way
# model.save('image_classifier.h5')

