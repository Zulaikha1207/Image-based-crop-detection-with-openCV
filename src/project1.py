import numpy as np
import argparse
import cv2

class CropDetector:
    def __init__(self, image):
        self.image = image

    def preprocess_image(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        preprocessed_image = cv2.threshold(blurred, 0, 250, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return preprocessed_image

    def apply_morphological_operations_opening(preprocessed_image, kernelSize):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernelSize)
        opening = cv2.morphologyEx(preprocessed_image, cv2.MORPH_OPEN, kernel)
        return opening

    def apply_morphological_operations_closing(preprocessed_image, kernelSize):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernelSize)
        closing = cv2.morphologyEx(preprocessed_image, cv2.MORPH_CLOSE, kernel)
        return closing

    def detect_crops(self, closing):
        contours, _ = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        crops = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 130:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                crops.append((center, radius))
                cv2.circle(self.image, center, radius, (0, 255, 0), 2)
        
        result = cv2.putText(self.image, f"Crops Detected: {len(crops)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imwrite("output.png", result)
        cv2.imshow('Detected Crops', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return result      