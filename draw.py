import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf

from examples.mnist import load_model

class DigitDrawerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Drawer")

        self.model = load_model()

        self.canvas_size = 280  # Size of the canvas (10x the 28x28 grid)
        self.cell_size = self.canvas_size // 28  # Size of each cell

        self.prediction_label = tk.Label(root, text="Predicted Number: ", font=("Helvetica", 24))
        self.prediction_label.pack()

        self.canvas = tk.Canvas(root, bg='white', width=self.canvas_size, height=self.canvas_size)
        self.canvas.pack()

        self.clear_button = tk.Button(root, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side='left')

        self.predict_button = tk.Button(root, text="Predict", command=self.predict)
        self.predict_button.pack(side='right')

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)

        self.image = np.zeros((28, 28))

    def paint(self, event):
        x, y = event.x, event.y
        x1, y1 = (x // self.cell_size) * self.cell_size, (y // self.cell_size) * self.cell_size
        x2, y2 = x1 + self.cell_size, y1 + self.cell_size
        self.canvas.create_rectangle(x1, y1, x2, y2, fill='black')
        self.image[y1//self.cell_size, x1//self.cell_size] = 1
    
        # Add less intense pixels to the neighboring cells
        neighbors = [(x1-self.cell_size, y1), (x1+self.cell_size, y1), (x1, y1-self.cell_size), (x1, y1+self.cell_size)]
        for nx, ny in neighbors:
            if 0 <= nx < self.canvas_size and 0 <= ny < self.canvas_size:
                if self.image[ny//self.cell_size, nx//self.cell_size] < 1:
                    self.canvas.create_rectangle(nx, ny, nx+self.cell_size, ny+self.cell_size, fill='#808080')
                    self.image[ny//self.cell_size, nx//self.cell_size] = 0.5
    
        self.predict()

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image.fill(0)
        self.prediction_label.config(text="Predicted Number: ")

    def predict(self):
        # Preprocess the image for the model
        input_image = self.image.reshape(1, 28, 28)
        input_image = input_image.astype('float32')
        
        # Get the model's prediction
        prediction = self.model.predict(input_image)
        predicted_digit = np.argmax(prediction)

        self.prediction_label.config(text=f"Predicted Number: {predicted_digit}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitDrawerApp(root)
    root.mainloop()