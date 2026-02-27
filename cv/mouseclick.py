import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt

# Define the image path using os.path for better cross-platform compatibility


# Load the image before defining the callback function
img = cv.imread("cat.png")
if img is None:
    raise FileNotFoundError(f"Could not load image from {"cat.py"}")

def click_event(event, x, y, flags, params):
    """
    Handle mouse click events on the image.
    
    Args:
        event: OpenCV mouse event type
        x: x-coordinate of mouse position
        y: y-coordinate of mouse position
        flags: Additional flags
        params: Additional parameters
    
    Features:
        - Left Click: Display and print (x,y) coordinates
        - Right Click: Display and print BGR color values at clicked position
    """
    text = ''
    font = cv.FONT_HERSHEY_COMPLEX
    color = (255, 0, 0)

    try:
        if event == cv.EVENT_LBUTTONDOWN:
            print(f"Coordinates: ({x}, {y})")
            text = f"{x},{y}"
            color = (0, 255, 0)  # Green
        elif event == cv.EVENT_RBUTTONDOWN:
            b = img[y, x, 0]
            g = img[y, x, 1]
            r = img[y, x, 2]
            print(f"BGR Values: ({b}, {g}, {r})")
            text = f"{b},{g},{r}"
            color = (0, 0, 255)  # Red

        cv.putText(img, text, (x, y), font, 0.5, color, 1, cv.LINE_AA)
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