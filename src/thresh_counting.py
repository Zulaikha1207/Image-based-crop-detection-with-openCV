import numpy as np
import argparse
import cv2

#construct argument parser and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default="crop1.png",
help="path to input image")
args = vars(ap.parse_args())

#load image and display it
image = cv2.imread(args["image"])
cv2.imshow("Original image", image)

# convert image to grayscale and threshold it
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (11, 11), 0)

thresh = cv2.threshold(blurred, 0, 250, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
cv2.imshow("Thresh", thresh)
cv2.waitKey(1000)

kernelSizes = [(3, 3), (5, 5), (7, 7)]
# loop over the kernels sizes
for kernelSize in kernelSizes:
	# construct a rectangular kernel from the current size and then
	# apply an "opening" operation
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
	opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
	cv2.imshow("Opening: ({}, {})".format(
		kernelSize[0], kernelSize[1]), opening)
	cv2.waitKey(1000)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
cv2.imshow("opening", opening)
cv2.waitKey(2000)

kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel2)
cv2.imshow("closing", closing)
cv2.waitKey(2000)


# find contours in the threshold image
contours,h = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
crop_count = 0
for contour in contours:
    # Calculate the area of the contour
    area = cv2.contourArea(contour)

    # Filter contours based on area
    if area > 130:
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