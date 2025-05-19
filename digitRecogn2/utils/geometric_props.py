import cv2
import numpy as np


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