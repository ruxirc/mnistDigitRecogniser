import cv2
import numpy as np
import torch

def run_inference(image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load("./models/m1.pth", map_location=device)
    model.eval()  # Set the model to evaluation mode

    # Redimensionare la 28x28 și normalizare
    img28 = cv2.resize(image, (28, 28))
    img28 = img28.astype(np.float32) / 255.0

    # Calcularea trăsăturilor geometrice
    geo_features = calculate_geometric_features(img28)

    # Scalare manuală a trăsăturilor geometrice (exemplu: min-max scaling)
    geo_scaled = (geo_features - np.min(geo_features)) / (np.max(geo_features) - np.min(geo_features))
    geo_scaled = geo_scaled.reshape(1, -1)

    # Convertire la tensori
    img_tensor = torch.tensor(img28).unsqueeze(0).unsqueeze(0).float().to(device)  # (1, 1, 28, 28)
    geo_tensor = torch.tensor(geo_scaled).float().to(device)  # (1, num_features)

    # Predicție cu modelul
    with torch.no_grad():
        outputs = model(img_tensor, geo_tensor)
        probs = torch.softmax(outputs, dim=1)
        prob_val, pred = torch.max(probs, 1)

    return pred.item(), prob_val.item()

# Functia pentru calcul geometric features
def calculate_geometric_features(image):
    image_uint8 = (image * 255).astype(np.uint8)
    moments = cv2.moments(image_uint8)

    area = moments['m00']
    cx = moments['m10'] / area if area != 0 else 0
    cy = moments['m01'] / area if area != 0 else 0

    Ixx = moments['mu20'] / area if area != 0 else 0
    Iyy = moments['mu02'] / area if area != 0 else 0
    Ixy = moments['mu11'] / area if area != 0 else 0

    angle = 0.5 * np.arctan2(2 * Ixy, Ixx - Iyy)

    contours, _ = cv2.findContours(image_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = cv2.arcLength(contours[0], True) if contours else 0

    thinness = (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else 0

    x, y, w, h = cv2.boundingRect(image_uint8)
    elongation = h / w if w != 0 else 0

    return [area, cx, cy, angle, perimeter, thinness, elongation]