import cv2
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default="shape.png",
    help="path to input image")
args = vars(ap.parse_args())

# Read image and display it
image = cv2.imread(args["image"])

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Perform Canny edge detection
edges = cv2.Canny(blurred, 50, 150)

# Find contours in the edge map
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw circles around the detected crops and count them
crop_count = 0
for contour in contours:
    # Calculate the area of the contour
    area = cv2.contourArea(contour)

    # Filter contours based on area
    if area > 1000:
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