import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv.imread("lines.png")        #Cat

if img is None:
    print("Image not found!")
    exit()

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# -------- Sobel Kernels --------
sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

sobel_y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=np.float32)

# -------- Apply Convolution --------
grad_x = cv.filter2D(gray, cv.CV_32F, sobel_x)
grad_y = cv.filter2D(gray, cv.CV_32F, sobel_y)

# Convert to absolute values
grad_x = np.absolute(grad_x)
grad_y = np.absolute(grad_y)

grad_x = np.uint8(grad_x)
grad_y = np.uint8(grad_y)

# -------- Combine Both --------
combined_edges = cv.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)

# -------- Display Results --------
plt.figure(figsize=(12,5))

plt.subplot(1,3,1)
plt.title("Vertical Edges (Sobel X)")
plt.imshow(grad_x, cmap="gray")
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Horizontal Edges (Sobel Y)")
plt.imshow(grad_y, cmap="gray")
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Combined Edges")
plt.imshow(combined_edges, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()