import tflite_runtime.interpreter as tflite
import numpy as np
import cv2
import tkinter as tk
from PIL import Image, ImageTk
from threading import Thread
import os

# Set Parameters
SEGMENTATION_MODEL_PATH = "models/model1.tflite"
CLASSIFICATION_MODEL_PATH = "models/model2.tflite"
TEST_PREDICT_DIR = 'datasets/test_predict'

CLASS_NAMES = ['immature', 'mature', 'normal']

interpreter_segmentation = tflite.Interpreter(SEGMENTATION_MODEL_PATH)
interpreter_classification = tflite.Interpreter(CLASSIFICATION_MODEL_PATH)

image_path = []
for root, dirs, files in os.walk(TEST_PREDICT_DIR):
    for file in files:
        path_og = os.path.join(root,file)
        image_path.append(path_og)

interpreter_segmentation.allocate_tensors()
interpreter_classification.allocate_tensors()

_, height, width, _ = interpreter_segmentation.get_input_details()[0]['shape']

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

    print(path)
    print(output_classification)