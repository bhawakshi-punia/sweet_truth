import os
from functools import lru_cache

import cv2
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.image import imread

from sweet_truth import image as st_image
from sweet_truth import text as st_text
from sweet_truth.errors import DetectionError
from sweet_truth.ocr import OcrEngine

# These are the functions used by the notebooks. The actual work now lives in the
# sweet_truth package (src/sweet_truth), which the web app and the CLI share;
# the functions here keep their original names, arguments and outputs.


# spaCy and Tesseract are slow to load, so they are loaded once and reused
@lru_cache(maxsize=1)
def _nlp():
    return st_text.load_nlp()


@lru_cache(maxsize=1)
def _ocr():
    return OcrEngine()


def _resources(model):
    return st_text.Resources(nlp=_nlp(), model=model, ocr=_ocr())


# Preprocessing function to
# Remove noise (e.g., text in brackets, special characters)
# Normalize case and remove leading/trailing spaces
def preprocess_text(text):
    cleaned, _ = st_text._strip_noise(text)
    doc = _nlp()(cleaned.lower().strip())
    # Lemmatize and remove stop words as well
    return " ".join([token.lemma_ for token in doc if not token.is_stop])


# to present the detected sweeteners as a dataframe
def _sweeteners_dataframe(sweeteners):
    if not sweeteners:
        return "No sweeteners detected."
    df = pd.DataFrame(
        {
            "Ingredient": [item.normalized for item in sweeteners],
            "Confidence": [item.confidence * 100 for item in sweeteners],
        }
    )
    df.index = df.index + 1  # i want the dataframe index to start from 1 and not 0
    return df


# generating a function to analyse the ingredient list, predict which of them is a sweetener using any classification model
# and display the results in a dataframe
def sweet_truth_from_list(lst, model):
    """
    This function generates a dataframe presenting each ingredient predicted as a sweetener
    and how confident our model is on the prediction.
    """
    result = st_text.classify_ingredients(lst, _resources(model))
    return _sweeteners_dataframe(result.sweeteners)


# to preprocess the image - noise removal, grayscaling, etc.
# (kept for the OCR notebook; the detector itself now gives Tesseract the plain grayscale
# image, which read the sample labels better than this fixed threshold)
def preprocess_image(image_path):
    # Load the image in grayscale
    img = cv2.imread(image_path, 0)
    # Apply Gaussian blur to remove noise
    img = cv2.GaussianBlur(img, (5, 5), 0)
    # Thresholding for better text recognition
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    return img


def _show_image(image_path):
    plt.imshow(imread(image_path))
    plt.axis("off")
    plt.show()


def _detect(image_path, model):
    with open(image_path, "rb") as f:
        return st_image.detect_from_image(f.read(), _resources(model))


# The second function - sweetener detection from an image - output is a dataframe of detected sweeteners
def sweet_truth_from_image(image_path, model):
    """
    Output is a dataframe with detected sweeteners along with the confidence in the prediction.
    """
    # Step 1: show the original image
    _show_image(image_path)

    # Step 2: perform OCR on the image, pick out the ingredient list and run our detector
    try:
        result = _detect(image_path, model)
    except DetectionError:
        return print("The image is not clear enough for appropriate text extraction.")
    return _sweeteners_dataframe(result.sweeteners)


# This is a function which will help us combine the bounding boxes around the sweeteners detected in the image.
def combine_boxes(boxes, x_threshold=40, y_threshold=20):
    """
    Combine horizontally aligned bounding boxes based on a threshold.

    Args:
        boxes (list of tuples): List of bounding boxes as (x1, y1, x2, y2).
        x_threshold (int): Maximum gap allowed between boxes on the x-axis.
        y_threshold (int): Maximum difference in y-coordinates to consider boxes on the same line.

    Returns:
        list of tuples: Combined bounding boxes.
    """
    if not boxes:
        return []

    # Sort boxes by x (left) coordinate and then by y (top) coordinate
    boxes = sorted(boxes, key=lambda b: (b[0], b[1]))

    combined_boxes = []
    current_box = boxes[0]

    for box in boxes[1:]:
        # Check if the boxes are on the same line and close enough on the x-axis
        if (
            abs(current_box[1] - box[1]) < y_threshold
            and abs(box[0] - current_box[2]) < x_threshold
        ):
            # Merge the current box with the new box
            current_box = (
                min(current_box[0], box[0]),  # x1
                min(current_box[1], box[1]),  # y1
                max(current_box[2], box[2]),  # x2
                max(current_box[3], box[3]),  # y2
            )
        else:
            # Add the current box to the result and start a new one
            combined_boxes.append(current_box)
            current_box = box

    # Add the last box
    combined_boxes.append(current_box)
    return combined_boxes


# To output an annotated image with the sweeteners detected along with the dataframe
def annotate_image_with_sweeteners(image_path, model):
    """
    Output is the annotated image and a dataframe with detected sweeteners along with the confidence in the prediction.
    """
    # showing the original image
    _show_image(image_path)

    # Steps 1-3: perform OCR, sweetener detection and locate the sweeteners in the image
    try:
        result = _detect(image_path, model)
    except DetectionError:
        print("The image is not clear enough for appropriate text extraction.")
        return None

    if not result.sweeteners:
        print("No sweeteners detected.")
        return None

    # Step 4: Draw rectangles around detected sweeteners on the original image (in color)
    original_img = cv2.imread(image_path)
    for box in result.boxes:
        cv2.rectangle(original_img, (box.x1, box.y1), (box.x2, box.y2), (0, 0, 255), 3)

    # to save the annotated image now
    base = os.path.basename(image_path)
    base_name = os.path.splitext(base)[0]
    output_path = os.path.join(os.path.dirname(image_path), f"{base_name}_annotated.jpg")
    cv2.imwrite(output_path, original_img)

    # displaying the annotated image
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")  # Hide axes
    plt.title("Annotated Image with Detected Sweeteners")
    plt.show()

    return _sweeteners_dataframe(result.sweeteners)
