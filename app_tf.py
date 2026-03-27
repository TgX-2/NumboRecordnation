import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
from tensorflow import keras
import os

base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, "model.h5")
model = keras.models.load_model(model_path)


WIDTH = 500
HEIGHT = 500

root = tk.Tk()
root.title("Nận giện trữ xố")

canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="white")
canvas.pack()

image = Image.new("L", (WIDTH, HEIGHT), 0)
draw = ImageDraw.Draw(image)

def paint(event):
    x1, y1 = event.x - 15, event.y - 15
    x2, y2 = event.x + 15, event.y + 15
    canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")
    draw.ellipse([x1, y1, x2, y2], fill=255)

canvas.bind("<B1-Motion>", paint)

def predict():
    img = np.array(image)

    coords = np.column_stack(np.where(img > 50))
    if coords.size == 0:
        label.config(text="Viết gì đi chứ?")
        return

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    img = img[y_min:y_max+1, x_min:x_max+1]

    img = Image.fromarray(img)
    img = img.resize((20, 20))

    new_img = Image.new("L", (28, 28), 0)
    new_img.paste(img, (4, 4))

    img = np.array(new_img)

    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)

    pred = model.predict(img, verbose=0)
    digit = np.argmax(pred)
    confidence = np.max(pred)

    label.config(text=f"{digit} ({confidence:.2f})")


def clear():
    canvas.delete("all")
    draw.rectangle([0, 0, WIDTH, HEIGHT], fill=0)
    label.config(text="Viết gì đi chứ?")


btn_predict = tk.Button(root, text="Dự đoán", command=predict)
btn_predict.pack()

btn_clear = tk.Button(root, text="Xóa hết đi", command=clear)
btn_clear.pack()

label = tk.Label(root, text="Viết gì đi chứ?", font=("Arial", 16))
label.pack()

root.mainloop()