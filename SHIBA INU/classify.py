import numpy as np
import tensorflow as tf

# Load the model
interpreter = tf.lite.Interpreter(model_path="model_quantized.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load image
image = tf.keras.preprocessing.image.load_img("image_to_classify1.jpg", target_size=(128, 128))
input_image = tf.keras.preprocessing.image.img_to_array(image)
input_image = np.expand_dims(input_image, axis=0)

# Normalize the input image
input_image = input_image / 255.0

# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], input_image)

# Run the model
interpreter.invoke()

# Get the output tensor
output_data = interpreter.get_tensor(output_details[0]['index'])

# Get the class prediction
if output_data[0][0] > 0.5:
    print("This is Shiba")
else:
    print("This is not Shiba")


# Load image
image1 = tf.keras.preprocessing.image.load_img("image_to_classify2.jpg", target_size=(128, 128))
input_image1 = tf.keras.preprocessing.image.img_to_array(image1)
input_image1 = np.expand_dims(input_image1, axis=0)

# Normalize the input image
input_image1 = input_image1 / 255.0

# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], input_image1)

# Run the model
interpreter.invoke()

# Get the output tensor
output_data = interpreter.get_tensor(output_details[0]['index'])

# Get the class prediction
if output_data[0][0] > 0.5:
    print("This is Shiba")
else:
    print("This is not Shiba")

