import pytesseract
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np


# Preprocess the image to improve OCR accuracy
def preprocess_image(image):
    # Convert to grayscale
    gray_image = image.convert('L')
    # Convert the PIL image to an OpenCV image
    open_cv_image = np.array(gray_image)

    # Apply GaussianBlur to remove noise
    blurred = cv2.GaussianBlur(open_cv_image, (5, 5), 0)

    # Apply adaptive thresholding for better binarization
    thresh_img = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Convert back to PIL format
    pil_img = Image.fromarray(thresh_img)

    # Optionally sharpen the image to further enhance clarity
    enhanced_image = pil_img.filter(ImageFilter.SHARPEN)
    enhancer = ImageEnhance.Contrast(enhanced_image)
    contrast_image = enhancer.enhance(2)  # Increase contrast

    return contrast_image


# Extract text from the PDF with preprocessing and OCR settings
def extract_text_from_pdf(pdf_path, start_page=112, end_page=118, dpi=300):
    text = ""

    # Convert pages to images
    images = convert_from_path(pdf_path, first_page=start_page + 1, last_page=end_page, dpi=dpi)

    for img in images:
        # Preprocess the image
        preprocessed_img = preprocess_image(img)
        # OCR the preprocessed image using pytesseract with custom configuration
        custom_config = r'--oem 3 --psm 3'  # OEM 3 uses LSTM engine, PSM 3 is fully automatic page segmentation
        page_text = pytesseract.image_to_string(preprocessed_img, config=custom_config)
        text += page_text

    return text


# Specify your PDF path
pdf_path = ''
# Extract text from the first 40 pages with high DPI and preprocessing
text_output = extract_text_from_pdf(pdf_path, start_page=112, end_page=130)

# Print or save the extracted text
if text_output:
    with open(f'{pdf_path}_extracted_text.txt', mode="w", encoding="utf-8") as file:
        file.write(text_output)

print("Text extraction completed.")
