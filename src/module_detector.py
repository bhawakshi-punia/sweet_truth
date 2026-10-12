import spacy
import re
import numpy as np
import pandas as pd 
import cv2
import pytesseract 
from matplotlib.image import imread
import matplotlib.pyplot as plt
import os 


# Preprocessing function to
# Remove noise (e.g., text in brackets, special characters)
# Normalize case and remove leading/trailing spaces
def preprocess_text(text):

    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")
    
    # the ingredient list may involve text in the form "abc (def (ghi))" or abc [d (e) f]
    # main idea is i wish to remove all kinds text in brackets or any full stops.
    pattern = r'\([^()]*\)|\[[^\[\]]*\]|[.]'
    
    # Continue removing brackets until no nested brackets remain
    while re.search(pattern, text):
        text = re.sub(pattern, ',', text)
    
    text = text.lower().strip()  
    doc = nlp(text)
    # Lemmatize and remove stop words as well
    return " ".join([token.lemma_ for token in doc if not token.is_stop])

# generating a function to analyse the ingredient list, predict which of them is a sweetener using any classification model 
# and display the results in a dataframe

def sweet_truth_from_list(lst, model):    
    '''
    This function generates a dataframe presenting each ingredient separately, 
    our model's prediction whether its a sweetener or not and how confident our model is 
    on the prediction. 
    '''
    
    # Step 1: to preprocess the list - remove noise, leading/trailing spaces and normalise text case
    prep_list = preprocess_text(lst)
    
    # Step 2: tokenize the list
    rx = re.compile(r"\s*(?:\b(?:and|or)\b|[,:])\s*") 
    tokens = rx.split(prep_list)

    # Step 3: Predicting for each item in list and storing them 
    result = []    
    for _, txt in enumerate(tokens): 
    
        prediction = 'Non-sweetener' if model.predict([txt])[0] == 0 else 'Sweetener'
        prob = max(model.predict_proba([txt])[0])*100
        
        result.append({'Ingredient': txt, 'Prediction': prediction, 'Confidence': prob})

    # Step 4: Preparing the results as a dataframe
    df = pd.DataFrame(result)

    # 4.1: extracting all entries that are predicted as a sweetener
    df2 = df[df['Prediction']=='Sweetener']
    df2 = df2.reset_index()
    df2.index = df2.index + 1 # i want the dataframe index to start from 1 and not 0
    df2 = df2.drop(['index', 'Prediction'], axis=1)  
    
    if len(df2)!=0: 
        return df2
    else: 
        return "No sweeteners detected."  

# to preprocess the image - noise removal, grayscaling, etc.
def preprocess_image(image_path):
    # Load the image in grayscale
    img = cv2.imread(image_path, 0)
    # Apply Gaussian blur to remove noise
    img = cv2.GaussianBlur(img, (5, 5), 0)
    # or median blur to address salt-and-pepper noise 
    #img = cv2.medianBlur(img,5)
    # Thresholding for better text recognition
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    return img
        
# The second function - sweetener detection from an image - output is a dataframe of detected sweeteners

import pytesseract 
import re 

def sweet_truth_from_image(image_path, model): 
    '''
    Output is a dataframe with detected sweeteners along with the confidence in the prediction.
    '''

    # Step 1: show the original image 
    plt.imshow(imread(image_path)) 
    plt.axis("off")    
    plt.show()
    
    # Step 2: preprocess the image - noise removal, grayscaling, etc.
    # we use our function preprocess_image(path) for this 
    img = preprocess_image(image_path)

    # Step 3: performing OCR on the preprocessed image 'img'
    extracted_text = pytesseract.image_to_string(img)
    #print(extracted_text)

    extracted_text = re.sub(r'\s+', ' ', extracted_text.strip())

    # this is an additional step, if by any chance, the OCR wasn't successful due to possible poor image resolution. 
    ing_list = None
    
    # Step 4: picking out the ingredient section from this extracted text
    ings = re.findall(r'ingredients.*', extracted_text, re.IGNORECASE)  # picking out the ingredient list from the image
    if ings != []:
            ing_list = ings[0] # this is our desired ingredient list we needed from the image

    if ing_list:
        # Step 5: preprocessing this list so that it is aligned with the desired input format for our detector 
        rx = re.compile(r"\s*(?:\b(?:and|or)\b|[,:])\s*") 
        rev_lst = rx.split(ing_list)
        
        result_string = ", ".join(rev_lst)
    
        # finally, comes our detector now 
        return sweet_truth_from_list(result_string, model)
    else: 
        return print("The image is not clear enough for appropriate text extraction.")

# To output an annotated image with the sweeteners detected along with the dataframe 

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
    # Sort boxes by y (top) coordinate and then by x (left) coordinate
    boxes = sorted(boxes, key=lambda b: (b[0], b[1]))

    #print(boxes)

    combined_boxes = []
    current_box = boxes[0]

    for box in boxes[1:]:
        # Check if the boxes are on the same line (y-coordinate proximity)
        if abs(current_box[1] - box[1]) < y_threshold:
            # Check if the boxes are close enough on the x-axis
            if abs(box[0] - current_box[2]) < x_threshold:
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
            #print("x_threshold loop", current_box)
        else:
            # Add the current box to the result and start a new one
            combined_boxes.append(current_box)
            current_box = box
            #print("y_threshold loop", current_box)

    # Add the last box
    combined_boxes.append(current_box)
    return combined_boxes

# now the function which annotates the image 

import re

def annotate_image_with_sweeteners(image_path, model):
    '''
    Output is the annotated image and a dataframe with detected sweeteners along with the confidence in the prediction.
    '''

    def clean_text(text):
        return re.sub(r'[^a-zA-Z0-9]', ' ', text.lower().strip())
    
    # showing the original image 
    plt.imshow(imread(image_path)) 
    plt.axis('off') 
    plt.show()
    
    # Step 1: Preprocess the image
    img = preprocess_image(image_path)
    
    # Step 2: Extract bounding box data using Pytesseract
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    
    # Load the original image in color (for annotation)
    original_img = cv2.imread(image_path)
    
    # Step 3: Perform OCR and sweetener detection
    extracted_text = pytesseract.image_to_string(img)

    extracted_text = re.sub(r'\s+', ' ', extracted_text.strip())

    # this is an additional step, if by any chance, the OCR wasn't successful due to possible poor image resolution. 
    ing_list = None
    
    # Step 4: picking out the ingredient section from this extracted text and annotating the image
    ings = re.findall(r'ingredients.*', extracted_text, re.IGNORECASE)  # picking out the ingredient list from the image
    if ings != []:
            ing_list = ings[0] # this is our desired ingredient list we needed from the image

    if ing_list:
        # Preprocess the extracted ingredient list
        rx = re.compile(r"\s*(?:\b(?:ingredient|ingredients|and|or)\b|[,:])\s*") 
        #rx = re.compile(r"\s*(?:\b(?:ingredient|ingredients|and|or)\b|[,:]|\s+)\s*")
        rev_lst = rx.split(ing_list)
        result_string = ", ".join(rev_lst)
        
        # Detect sweeteners
        sweeteners_df = sweet_truth_from_list(result_string, model)
        
        if isinstance(sweeteners_df, pd.DataFrame):
            # Step 4: Draw rectangles around detected sweeteners
            detected_sweeteners = []
            for ingredient in sweeteners_df['Ingredient'].str.lower().tolist(): 
                for t in ingredient.split():
                    detected_sweeteners.append(t)

            n_boxes = len(data['text'])
            detected_boxes = []
            for i in range(n_boxes):
                # Clean the OCR word
                word = clean_text(data['text'][i])
            
                # Skip empty or invalid words
                if not word or len(word) < 3:
                    continue
            
                # Check for bidirectional substring matching with noise removal
                if any(word in sweetener or sweetener in word for sweetener in detected_sweeteners):
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    entry = (x, y, x + w, y + h)
                    detected_boxes.append(entry)
            
            # Combine nearby boxes
            combined_boxes = combine_boxes(detected_boxes)
            
            # Draw combined boxes on the image
            for (x1, y1, x2, y2) in combined_boxes:
                cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 0, 255), 3)

            
            annotated_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)            

            # to save the annotated image now 
            base = os.path.basename(image_path)
            base_name = os.path.splitext(base)[0]
            output_path = os.path.join(os.path.dirname(image_path), f"{base_name}_annotated.jpg")
            cv2.imwrite(output_path, annotated_img_rgb)

            # displaying the annotated image
            plt.imshow(annotated_img_rgb)
            plt.axis('off')  # Hide axes
            plt.title("Annotated Image with Detected Sweeteners")
            plt.show()

            return sweeteners_df
        else:
            print("No sweeteners detected.")
    else:
        print("The image is not clear enough for appropriate text extraction.")
