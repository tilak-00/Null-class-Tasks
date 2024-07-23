import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

# Load pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

def predict_animal(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=1)[0]
    _, class_label, confidence = decoded_predictions[0]
    return class_label, confidence


def browse_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        animal, confidence = predict_animal(file_path)
        result_label.config(text=f"Predicted animal: {animal} (Confidence: {confidence:.2f})")
        display_image(file_path)

def display_image(image_path):
    img = Image.open(image_path)
    img.thumbnail((300, 300))
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img

# Create the main window
root = tk.Tk()
root.title("Animal Prediction GUI")

# Create widgets
browse_button = tk.Button(root, text="Browse Image", command=browse_image)
result_label = tk.Label(root, text="")
image_label = tk.Label(root)

# Pack widgets
browse_button.pack(pady=10)
result_label.pack()
image_label.pack()

root.mainloop()
