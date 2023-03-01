import numpy as np
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model("model.h5")

# Load image
image = tf.keras.preprocessing.image.load_img("image_to_classify1.jpg", target_size=(64, 64))
input_image = tf.keras.preprocessing.image.img_to_array(image)
input_image = np.expand_dims(input_image, axis=0)

# Normalize the input image
input_image = input_image / 255.0

# Get the class prediction
output_data = model.predict(input_image)
if output_data[0][0] > 0.5:
    print("This is Shiba")
else:
    print("This is not Shiba")


# Load image
image1 = tf.keras.preprocessing.image.load_img("image_to_classify2.jpg", target_size=(64, 64))
input_image1 = tf.keras.preprocessing.image.img_to_array(image1)
input_image1 = np.expand_dims(input_image1, axis=0)

# Normalize the input image
input_image1 = input_image1 / 255.0

# Get the class prediction
output_data = model.predict(input_image1)
if output_data[0][0] > 0.5:
    print("This is Shiba")
else:
    print("This is not Shiba")

