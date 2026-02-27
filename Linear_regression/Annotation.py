import cv2 as cv
import os

# -------- CONFIG --------
image_path = "image.png"
class_id = 0   # change class id if needed
# ------------------------

img = cv.imread(image_path)
clone = img.copy()

height, width = img.shape[:2]

drawing = False
ix, iy = -1, -1
boxes = []

# Mouse callback function
def draw_bbox(event, x, y, flags, param):
    global ix, iy, drawing, img, boxes

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            img = clone.copy()
            cv.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)

    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        cv.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
        boxes.append((ix, iy, x, y))

cv.namedWindow("Annotation Tool")
cv.setMouseCallback("Annotation Tool", draw_bbox)

while True:
    cv.imshow("Annotation Tool", img)
    key = cv.waitKey(1) & 0xFF

    # Press 's' to save annotations
    if key == ord('s'):
        label_path = image_path.replace(".png", ".txt")
        with open(label_path, "w") as f:
            for (x1, y1, x2, y2) in boxes:
                # Convert to YOLO format
                x_center = ((x1 + x2) / 2) / width
                y_center = ((y1 + y2) / 2) / height
                box_width = abs(x2 - x1) / width
                box_height = abs(y2 - y1) / height

                f.write(f"{class_id} {x_center} {y_center} {box_width} {box_height}\n")

        print("Annotations saved!")
    
    # Press 'r' to reset
    elif key == ord('r'):
        img = clone.copy()
        boxes = []
        print("Reset done!")

    # Press 'q' to quit
    elif key == ord('q'):
        break

cv.destroyAllWindows()