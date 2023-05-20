import cv2
import argparse
import numpy as np

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default="shape.png",
    help="path to input image")
args = vars(ap.parse_args())

# Read image and display it
image = cv2.imread(args["image"])
image = cv2.resize(image, (750, 500))
cv2.imshow("resized", image)
cv2.waitKey(1000)

# convert to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow("hsv image", hsv)
cv2.waitKey(1000)

# Converting it to hue saturation value image
range1=(26,0,0)
range2=(86,255,255)
mask1=cv2.inRange(hsv,range1,range2)
cv2.imshow("hsv image", mask1)
cv2.waitKey(1000)

kernel1_size = (7, 7)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel1_size)
opened_mask = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel)
closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel)
cv2.imshow("closed image", closed_mask)
cv2.waitKey(1000)


contours,h = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
crop_count = 0
for contour in contours:
    # Calculate the area of the contour
    area = cv2.contourArea(contour)

    # Filter contours based on area
    if area > 150:
        # Get the center coordinates and radius of the enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)

        # Draw a circle around the crop
        cv2.circle(image, center, radius, (0, 255, 0), 2)

        # Increment the crop count
        crop_count += 1

# Display the image with crop count
cv2.putText(image, f"Crops Detected: {crop_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imshow('Detected Crops', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print the total crop count
print(f"Total Crops: {crop_count}")