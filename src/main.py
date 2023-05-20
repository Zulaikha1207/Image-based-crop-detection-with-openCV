import argparse
import cv2
from project1 import CropDetector


def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", type=str, default="shape.png",
        help="path to input image")
    args = vars(ap.parse_args())

    # Read image and create ImageProcessor instance
    image = cv2.imread(args["image"])
    image = CropDetector(image)
    preprocessed_image = CropDetector.preprocess_image(image)

    kernelSizes = [(3, 3), (5, 5), (7, 7)]
    for kernelSize in kernelSizes:
        opening = CropDetector.apply_morphological_operations_opening(preprocessed_image, kernelSize)
        cv2.imshow("Opening: ({}, {})".format(kernelSize[0], kernelSize[1]), opening)
        cv2.waitKey(1000)

    opening = CropDetector.apply_morphological_operations_opening(preprocessed_image, (7, 7))
    cv2.imshow("Opening", opening)
    cv2.waitKey(2000)

    closing = CropDetector.apply_morphological_operations_closing(opening, (7, 7))
    cv2.imshow("Closing", closing)
    cv2.waitKey(2000)

    result = CropDetector.detect_crops(image, closing)

if __name__ == "__main__":
    main()
