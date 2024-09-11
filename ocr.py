import pytesseract
import cv2
import re
import spacy

nlp = spacy.load("en_core_web_sm")


# Function to preprocess the image (optional but recommended for better OCR performance)
def preprocess_image(image_path):
    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Apply GaussianBlur to remove noise
    img = cv2.GaussianBlur(img, (5, 5), 0)
    # Apply thresholding to convert the image to binary format
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return img


# Function to perform OCR and extract text from the image
def extract_text_from_image(image_path):
    # Preprocess the image (you can skip this if your images are already clean)
    preprocessed_img = preprocess_image(image_path)
    # Use pytesseract to extract text
    text = pytesseract.image_to_string(preprocessed_img)
    return text


# Function to clean the OCR-extracted text
def clean_text(text):
    # This are some example that you can use to clean the text
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'[\x0c]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# Function to perform Named Entity Recognition using spaCy
def perform_ner(text):
    doc = nlp(text)
    # Extract entities recognized by spaCy
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    return entities


image_path = 'path_to_your_scanned_image.jpg'
extracted_text = extract_text_from_image(image_path)
print("Raw Extracted Text:\n", extracted_text)
cleaned_text = clean_text(extracted_text)
print("\nCleaned Text:\n", cleaned_text)
entities = perform_ner(cleaned_text)
print("\nNamed Entities:\n")
for entity, label in entities:
    print(f"{entity} ({label})")
