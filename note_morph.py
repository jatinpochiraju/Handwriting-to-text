import cv2
import pytesseract
import streamlit as st
from PIL import Image
import numpy as np

# Optional: set Tesseract path if needed
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

st.title("📝 NoteMorph AI - Handwritten Text to Digital")

uploaded_file = st.file_uploader("Upload an image of your handwritten notes", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to grayscale and preprocess
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 4)

    # OCR using Tesseract with handwritten config
    custom_config = r'--oem 1 --psm 6'
    text = pytesseract.image_to_string(thresh, config=custom_config)

    st.subheader("📄 Extracted Text:")
    st.text_area("", text, height=300)
