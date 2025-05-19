import torch
from torchvision import datasets

import numpy as np
import cv2

import tkinter as tk

from digitRecogn2.utils.geometric_props import calculate_geometric_features
from digitRecogn2.modelArchs.model_arch import CNNGeoModel

label_map = datasets.MNIST(root='./data', download=True).classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DrawingApp:
    def __init__(self, root, model, scaler):
        self.root = root
        self.model = model
        self.scaler = scaler
        self.model.eval()

        self.root.title("Desenează o cifră")

        self.canvas = tk.Canvas(root, width=280, height=280, bg='white')
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.paint)

        self.image = np.zeros((280, 280), dtype=np.uint8)
        self.drawing = False

        self.predict_button = tk.Button(root, text="Recunoaște", command=self.recognize)
        self.predict_button.pack()

        self.clear_button = tk.Button(root, text="Șterge", command=self.clear)
        self.clear_button.pack()

        self.prediction_label = tk.Label(root, text="Cifra prezisă: -", font=("Helvetica", 16))
        self.prediction_label.pack()

        self.accuracy_label = tk.Label(root, text="Probabilitate: -", font=("Helvetica", 14))
        self.accuracy_label.pack()

    def paint(self, event):
        x, y = event.x, event.y
        r = 8
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill='black')
        if 0 <= x < 280 and 0 <= y < 280:
            self.image[y-r:y+r, x-r:x+r] = 255
            self.drawing = True

    def clear(self):
        self.canvas.delete("all")
        self.image.fill(0)
        self.prediction_label.config(text="Cifra prezisă: -")
        self.accuracy_label.config(text="Probabilitate: -")
        self.drawing = False

    def recognize(self):
        if not self.drawing:
            return

        # Redimensionez la 28x28 si normalizez
        img28 = cv2.resize(self.image, (28, 28))
        img28 = img28.astype(np.float32) / 255.0

        # Calculez trăsături geometrice si scalez
        geo = np.array(calculate_geometric_features(img28)).reshape(1, -1)
        # geo_scaled = self.scaler.transform(geo)

        # Convert la tensor torch
        img_tensor = torch.tensor(img28).unsqueeze(0).unsqueeze(0).float().to(device)  # (1,1,28,28)
        geo_tensor = torch.tensor(geo).float().to(device)  # (1,7)

        with torch.no_grad():
            outputs = self.model(img_tensor, geo_tensor)
            probs = torch.softmax(outputs, dim=1)
            prob_val, pred = torch.max(probs, 1)
            # pred = label_map[pred.item()]
            pred = pred.item()

        self.prediction_label.config(text=f"Cifra prezisă: {pred}")
        self.accuracy_label.config(text=f"Probabilitate: {prob_val.item()*100:.2f}%")
        self.drawing = False

model = CNNGeoModel(num_geo_features=7, num_classes=10)
model.load_state_dict(torch.load('./models/m1.pth', map_location=device))
model.to(device)

# Pentru a porni aplicația:
root = tk.Tk()
app = DrawingApp(root, model)
root.mainloop()