import cv2
import numpy as np
import os

def segment_image(path,
                  blur_kernel=(5,5),
                  blur_sigma=0,
                  bin_thresh=140,
                  close_kernel=(3,3),
                  row_thresh=10,
                  col_thresh=10,
                  min_row_height=10,
                  min_col_width=5,
                  output_dir="segs"):
    """
    Read image, apply Gaussian blur, binarize, apply closing, then segment into text rows and letters.
    Saves letter images into `output_dir` and returns list of segment info.

    Parameters:
    - path: image file path
    - blur_kernel, blur_sigma: Gaussian blur params
    - bin_thresh: threshold for binarization
    - close_kernel: structuring element size for closing (w,h)
    - row_thresh: minimum projection sum to consider as text row
    - col_thresh: minimum projection sum to consider as letter column
    - min_row_height: minimum pixel height for a row
    - min_col_width: minimum pixel width for a letter
    - output_dir: directory to save letter images
    """
    # Load image
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")

    # Prepare output directory
    os.makedirs(output_dir, exist_ok=True)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(img, blur_kernel, blur_sigma)

    # Binarize (invert: text=white)
    _, binary = cv2.threshold(blurred, bin_thresh, 255, cv2.THRESH_BINARY_INV)

    # Morphological closing to connect components
    kx, ky = close_kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
    processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Horizontal projection on processed to detect rows
    horiz_proj = np.sum(processed, axis=1)
    rows = []
    start = None
    for y, val in enumerate(horiz_proj):
        if val > row_thresh and start is None:
            start = y
        elif val <= row_thresh and start is not None:
            if y - start >= min_row_height:
                rows.append((start, y))
            start = None
    # handle last row
    if start is not None and img.shape[0] - start >= min_row_height:
        rows.append((start, img.shape[0]))

    segments = []
    # Vertical projection within each row to detect letters
    for i, (y0, y1) in enumerate(rows):
        row_img = processed[y0:y1, :]
        vert_proj = np.sum(row_img, axis=0)

        cols = []
        start = None
        for x, val in enumerate(vert_proj):
            if val > col_thresh and start is None:
                start = x
            elif val <= col_thresh and start is not None:
                if x - start >= min_col_width:
                    cols.append((start, x))
                start = None
        # handle last column
        if start is not None and img.shape[1] - start >= min_col_width:
            cols.append((start, img.shape[1]))

        letters = []
        for j, (x0, x1) in enumerate(cols):
            letter_img = row_img[:, x0:x1]
            # Pad to make square
            h, w = letter_img.shape
            size = max(h, w)
            pad_top = (size - h) // 2
            pad_bottom = size - h - pad_top
            pad_left = (size - w) // 2
            pad_right = size - w - pad_left
            letter_img_square = cv2.copyMakeBorder(letter_img, pad_top, pad_bottom, pad_left, pad_right,
                                                   borderType=cv2.BORDER_CONSTANT, value=0)
            letters.append(letter_img_square)
            # Save each letter (invert back to black text)
            out_name = f"row_{i:02d}_letter_{j:02d}.png"
            out_path = os.path.join(output_dir, out_name)
            cv2.imwrite(out_path, cv2.bitwise_not(letter_img_square))

        segments.append({
            'row_range': (y0, y1),
            'cols': cols,
            'row_image': row_img,
            'letters': letters
        })

    return segments

if __name__ == "__main__":
    image_path = "/digitRecogn2/images/img.png"
    output_dir = "../segs"

    # Example: adjust row_thresh and col_thresh as needed
    segments = segment_image(
        image_path,
        row_thresh=0,
        col_thresh=100,
        output_dir=output_dir
    )
    print(f"Extracted {len(segments)} rows and saved letters to '{output_dir}'")
