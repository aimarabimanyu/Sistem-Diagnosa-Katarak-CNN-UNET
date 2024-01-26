import tflite_runtime.interpreter as tflite
import numpy as np
import cv2
import tkinter as tk
from PIL import Image, ImageTk
from threading import Thread
import os
import time
import datetime

def inference_model(frame):
    CLASS_NAMES = ['immature', 'mature', 'normal']
    start_time = time.time()

    # Load TFLite Model and Allocate Tensors.
    interpreter_segmentation = tflite.Interpreter(model_path="models/model1.tflite")
    interpreter_segmentation.allocate_tensors()
    interpreter_classification = tflite.Interpreter(model_path="models/model2.tflite")
    interpreter_classification.allocate_tensors()

    # Get Height and Width
    _, height, width, _ = interpreter_segmentation.get_input_details()[0]['shape']

    # Load Image from Path and Resize
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (width, height)).astype(np.float32)
    image = image / 255.0

    # Add Batch Dimension
    input_image = np.expand_dims(image, axis=0)
    input_image = np.expand_dims(input_image, axis=-1)

    # Set Input and Invoke Segmentation Model
    interpreter_segmentation.set_tensor(interpreter_segmentation.get_input_details()[0]['index'], input_image)
    interpreter_segmentation.invoke()

    # Get Output
    output_segmentation = np.squeeze(interpreter_segmentation.get_tensor(interpreter_segmentation.get_output_details()[0]['index']))
    output_segmentation = np.where(output_segmentation > 0.6, 1, 0).astype(np.uint8)

    # Fuse Image with Segmentation Mask with Bitwise Operation
    multipy_image = cv2.resize(frame, (width, height))
    multipy_image = cv2.bitwise_and(multipy_image, multipy_image, mask=output_segmentation)
    multipy_image = cv2.cvtColor(multipy_image, cv2.COLOR_BGR2RGB)

    # Add Batch Dimension for Predicted Image
    input_image = np.expand_dims(multipy_image, axis=0).astype(np.float32)

    # Set Input and Invoke Classification Model
    interpreter_classification.set_tensor(interpreter_classification.get_input_details()[0]['index'], input_image)
    interpreter_classification.invoke()

    # Get Output
    output_classification = np.squeeze(interpreter_classification.get_tensor(interpreter_classification.get_output_details()[0]['index']))
    output_classification = CLASS_NAMES[np.argmax(output_classification)]
    end_time = time.time()

    total_time = end_time - start_time
    now = datetime.datetime.now()
    filename = now.strftime('%Y%m%d_%H%M%S')
    cv2.imwrite("hasil/pasien" + filename + "_" + output_classification + "_kom_" + str(total_time) + ".jpg", multipy_image)
    cv2.imwrite("hasil/pasien" + filename + "_" + output_classification + "_kom_" + str(total_time) + "_clean" + ".jpg", frame)

    return output_classification

def update_image():
    while True:
        # Read Frame and Condition from Camera
        ret, frame = cam.read()

        if ret:
            # Resize Frame to 480x320
            frame = cv2.resize(frame, (480, 320))

            # Convert CV Image to PIL Image
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Convert PIL Image to Tkinter Image
            image = ImageTk.PhotoImage(image)

            # Show Image in Label
            label.config(image=image)
            label.image = image

def capture_image():
    # Read Frame and Condition from Camera
    ret, frame = cam.read()

    if ret:
        # Inference Model
        result = inference_model(frame)

        # Resize Frame to 380x220
        frame = cv2.resize(frame, (380, 220))

        # Open New Window
        new_window = tk.Toplevel(window)
        new_label = tk.Label(new_window)
        new_label.pack()

        # Convert CV Image to PIL Image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Convert PIL Image to Tkinter Image
        image = ImageTk.PhotoImage(image)

        # Show Image in Label
        new_label.config(image=image)
        new_label.image = image

        # Show Prediction Result on Window
        result_label = tk.Label(new_window, text=str(result))
        result_label.pack()

# Build Tkinter Window
window = tk.Tk()
window.title("Model Inference App")
window.geometry("480x320")

# Build Label to Show Image
label = tk.Label(window)
label.pack()

# Build Button Frame
button_frame = tk.Frame(window)
button_frame.place(relx=0.5, rely=0.9, anchor='center')

# Build Capture and Stop Button
button_capture = tk.Button(button_frame, text="Capture", command=capture_image)
button_capture.pack(side="left", expand=True)

# Open Camera Connection
cam = cv2.VideoCapture(0)

# Build Thread to Update Image
thread = Thread(target=update_image)
thread.start()

# Mainloop Tkinter Window
window.mainloop()

# Close Camera Connection and Stop Script
cam.release()
os._exit(0)