import tflite_runtime.interpreter as tflite
import numpy as np
import cv2
import tkinter as tk
from PIL import Image, ImageTk
from threading import Thread
import os

def inference_model(frame):
    CLASS_NAMES = ['mild', 'normal', 'severe']

    # Load TFLite model and allocate tensors.
    interpreter_segmentation = tflite.Interpreter(model_path="models/model1.tflite")
    interpreter_segmentation.allocate_tensors()
    interpreter_classification = tflite.Interpreter(model_path="models/model2.tflite")
    interpreter_classification.allocate_tensors()

    # Get height and width
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
    output_segmentation = np.where(output_segmentation > 0.9, 1, 0).astype(np.uint8)

    # Fuse Image with Segmentation Mask with Bitwise Operation
    multipy_image = cv2.resize(frame, (width, height))
    multipy_image = cv2.bitwise_and(multipy_image, multipy_image, mask=output_segmentation)
    multipy_image = cv2.cvtColor(multipy_image, cv2.COLOR_BGR2RGB)

    cv2.imwrite("test.jpg", multipy_image)

    # Add Batch Dimension for Predicted Image
    input_image = np.expand_dims(multipy_image, axis=0).astype(np.float32)

    # Set Input and Invoke Classification Model
    interpreter_classification.set_tensor(interpreter_classification.get_input_details()[0]['index'], input_image)
    interpreter_classification.invoke()

    # Get Output
    output_classification = np.squeeze(interpreter_classification.get_tensor(interpreter_classification.get_output_details()[0]['index']))
    output_classification = CLASS_NAMES[np.argmax(output_classification)]

    return output_classification

def update_image():
    while True:
        # Membaca frame dari kamera
        ret, frame = cam.read()

        if ret:
            # Mengubah ukuran frame menjadi 480x320
            frame = cv2.resize(frame, (480, 320))

            # Mengubah gambar OpenCV menjadi gambar PIL
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Mengubah gambar PIL menjadi gambar Tkinter
            image = ImageTk.PhotoImage(image)

            # Menampilkan gambar di label
            label.config(image=image)
            label.image = image

def capture_image():
    # Membaca frame dari kamera
    ret, frame = cam.read()

    if ret:
        # Melakukan inferensi model
        result = inference_model(frame)

        # Mengubah ukuran frame menjadi 480x320
        frame = cv2.resize(frame, (380, 220))

        # Membuka jendela baru untuk menampilkan hasil
        new_window = tk.Toplevel(window)
        new_label = tk.Label(new_window)
        new_label.pack()

        # Mengubah gambar OpenCV menjadi gambar PIL
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Mengubah gambar PIL menjadi gambar Tkinter
        image = ImageTk.PhotoImage(image)

        # Menampilkan gambar di label baru
        new_label.config(image=image)
        new_label.image = image

        # Menampilkan hasil prediksi
        result_label = tk.Label(new_window, text=str(result))
        result_label.pack()

def stop_button():
    # Close the window
    window.destroy()

    # Release the camera connection
    cam.release()

    # Stop the script
    os._exit(0)

# Membuat jendela Tkinter
window = tk.Tk()
window.title("Model Inference App")
window.geometry("480x320")

# Membuat label untuk menampilkan gambar
label = tk.Label(window)
label.pack()

# Membuat frame untuk menampung tombol
button_frame = tk.Frame(window)
button_frame.place(relx=0.5, rely=0.9, anchor='center')

# Membuat tombol untuk capture dan stop
button_capture = tk.Button(button_frame, text="Capture", command=capture_image)
button_capture.pack(side="left", padx=5, expand=True)
button_stop = tk.Button(button_frame, text="Stop", command=stop_button)
button_stop.pack(side="right", padx=5, expand=True)

# Membuka koneksi ke kamera USB
cam = cv2.VideoCapture(0)

# Membuat thread untuk memperbarui gambar
thread = Thread(target=update_image)
thread.start()

# Menjalankan loop utama Tkinter
window.mainloop()

# Melepaskan koneksi ke kamera setelah jendela ditutup
cam.release()