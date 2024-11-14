import numpy as np
import cv2
from tensorflow.keras.models import load_model
import sys
sys.stdout.reconfigure(encoding='utf-8')

model = load_model("handwritten_digit_model.h5")

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found or unable to open.")
    img = cv2.resize(img, (28, 28))
    img = 255 - img
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img

def predict_digit(img_path):
    img = preprocess_image(img_path)
    prediction = model.predict(img)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction)
    return predicted_digit, confidence

image_path = "0/Zero_full (15).jpg"
# image_path = "1/One_full (7).jpg"
# image_path = "2/Two_full (1).jpg"
# image_path = "3/Three_full (1).jpg"
# image_path = "4/Four_full (1).jpg"
# image_path = "5/Five_full (1).jpg"
# image_path = "6/Six_full (23).jpg"
# image_path = "7/Seven_full (1).jpg"
# image_path = "8/Eight_full (1).jpg"
# image_path = "9/Nine_full (1).jpg"

digit, confidence = predict_digit(image_path)
print(f"Predicted Digit: {digit}, Confidence: {confidence:.2f}")
