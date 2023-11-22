import tflite_runtime.interpreter as tflite
import numpy as np
import cv2
import os

CLASS_NAMES = ['mild', 'normal', 'severe']

# Load TFLite model and allocate tensors.
interpreter_segmentation = tflite.Interpreter(model_path="models/model1.tflite")
interpreter_segmentation.allocate_tensors()
interpreter_classification = tflite.Interpreter(model_path="models/model2.tflite")
interpreter_classification.allocate_tensors()

# Get input and output tensors.
input_details_segmentation = interpreter_segmentation.get_input_details()
output_details_segmentation = interpreter_segmentation.get_output_details()
input_details_classification = interpreter_classification.get_input_details()
output_details_classification = interpreter_classification.get_output_details()

# Get height and width
_, height, width, _ = interpreter_segmentation.get_input_details()[0]['shape']

# Load image
image_path = []
for root, dirs, files in os.walk('test_predict/'):
    for file in files:
        path_og = os.path.join(root,file)
        image_path.append(path_og)

# Loop through all images
for path in image_path:
    # Load Image from Path and Resize
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (width, height)).astype(np.float32)
    image = image / 255.0

    # Add Batch Dimension
    input_image = np.expand_dims(image, axis=0)
    input_image = np.expand_dims(input_image, axis=-1)

    # Set Input and Invoke Segmentation Model
    interpreter_segmentation.set_tensor(interpreter_segmentation.get_input_details()[0]['index'], input_image)
    interpreter_segmentation.invoke()

    # Get Output
    output_segmentation = np.squeeze(
        interpreter_segmentation.get_tensor(interpreter_segmentation.get_output_details()[0]['index']))
    output_segmentation = np.where(output_segmentation > 0.9, 1, 0).astype(np.uint8)

    # Multiply Image with Mask
    multipy_image = cv2.imread(path)
    multipy_image = cv2.resize(multipy_image, (width, height))
    multipy_image = cv2.bitwise_and(multipy_image, multipy_image, mask=output_segmentation)
    multipy_image = cv2.cvtColor(multipy_image, cv2.COLOR_BGR2RGB)

    # Add Batch Dimension for Predicted Image
    input_image = np.expand_dims(multipy_image, axis=0).astype(np.float32)

    # Set Input and Invoke Classification Model
    interpreter_classification.set_tensor(interpreter_classification.get_input_details()[0]['index'], input_image)
    interpreter_classification.invoke()

    # Get Output
    output_classification = np.squeeze(
        interpreter_classification.get_tensor(interpreter_classification.get_output_details()[0]['index']))
    output_classification = CLASS_NAMES[np.argmax(output_classification)]
    print(output_classification)