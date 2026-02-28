import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt

# Define the image path using os.path for better cross-platform compatibility


# Load the image before defining the callback function
img = cv.imread("image1.png")
if img is None: 
    raise FileNotFoundError(f"Could not load image from {"cat.py"}")

def click_event(event, x, y, flags, params):
    """
    Handle mouse click events on the image.

    Left Click:
        - Display coordinates and RGB pixel values
    """

    font = cv.FONT_HERSHEY_COMPLEX

    try:
        if event == cv.EVENT_LBUTTONDOWN:

            # Get BGR values
            b, g, r = img[y, x]

            # Convert to RGB (just reorder)
            rgb_text = f"({x},{y}) RGB=({r},{g},{b})"

            print(f"Coordinates: ({x}, {y})  RGB: ({r}, {g}, {b})")

            # Optional: draw a small circle where clicked
            cv.circle(img, (x, y), 3, (0, 255, 0), -1)

            # Display text near the clicked point
            cv.putText(img,
                       rgb_text,
                       (x, y - 10),
                       font,
                       0.4,
                       (250, 250, 250),
                       1,
                       cv.LINE_AA)

            cv.imshow('image', img)

    except IndexError:
        print("Clicked outside image boundaries")

def main():
    cv.imshow('image', img)
    cv.setMouseCallback('image', click_event)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()