# Device
import cv2
import torch
from torchvision.transforms import transforms

from digitRecogn2.utils.geometric_props import calculate_geometric_features
from digitRecogn2.modelArchs.model_arch import CNNGeoModel
from digitRecogn2.utils.segmentation import segment_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
num_geo_features = 7
num_classes = 10
model = CNNGeoModel(num_geo_features=num_geo_features, num_classes=num_classes)
model.load_state_dict(torch.load('C:/UTCN/an3/sem2/pi/proj/mnistDigitRecogniser/digitRecogn2/models3/m1.pth', map_location=device))
model.to(device)
model.eval()

# MNIST normalization
normalize = transforms.Normalize((0.1307,), (0.3081,))
label_map = [str(i) for i in range(10)]

# Path to your input image
# img_path = "/digitRecogn2/images/5s.png"

def run_inference(img_path):
    # Segment into letters
    segments = segment_image(
        img_path,
        row_thresh=0,
        col_thresh=100,
        output_dir="segs"
    )

    res = []

    print(f"Found {len(segments)} text rows.")
    # Iterate rows and letters
    for i, seg in enumerate(segments):
        for j, letter_img in enumerate(seg['letters']):
            # Preprocess letter
            letter_resized = cv2.resize(letter_img, (28, 28), interpolation=None)
            arr = letter_resized.astype('float32') / 255.0
            letter_tensor = (
                torch.from_numpy(arr)
                     .unsqueeze(0)
                     .unsqueeze(0)
                     .to(device)
            )
            letter_tensor = normalize(letter_tensor)

            # Geometric features
            geo_feats   = calculate_geometric_features(letter_img)
            geo_tensor  = torch.tensor(geo_feats, dtype=torch.float32).unsqueeze(0).to(device)

            # Predict
            with torch.no_grad():
                outputs = model(letter_tensor, geo_tensor)
                pred_idx = outputs.argmax(dim=1).item()
                print(f"Row {i}, Letter {j} → {label_map[pred_idx]}")
                res.append(int(label_map[pred_idx]))

    return res