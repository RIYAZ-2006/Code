import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# -------- Load Image --------
img = cv.imread("image.png")

if img is None:
    print("Error: Image not found!")
    exit()

# Convert BGR to RGB for matplotlib
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# -------- 1️⃣ Blur Filter --------
blur_kernel = np.ones((3, 3), np.float32) / 9
blurred = cv.filter2D(img_rgb, -1, blur_kernel)

# -------- 2️⃣ Sharpen Filter --------
sharpen_kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])
sharpened = cv.filter2D(img_rgb, -1, sharpen_kernel)

# -------- 3️⃣ Edge Detection Filter --------
edge_kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])
edges = cv.filter2D(img_rgb, -1, edge_kernel)

# -------- 4️⃣ Custom Sobel Filter --------
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

sobel_kernel = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
])
sobel_edges = cv.filter2D(gray, -1, sobel_kernel)

# -------- Display All Results --------
plt.figure(figsize=(12, 8))

plt.subplot(2,3,1)
plt.title("Original")
plt.imshow(img_rgb)
plt.axis("off")

plt.subplot(2,3,2)
plt.title("Blur")
plt.imshow(blurred)
plt.axis("off")

plt.subplot(2,3,3)
plt.title("Sharpen")
plt.imshow(sharpened)
plt.axis("off")

plt.subplot(2,3,4)
plt.title("Edge Detection")
plt.imshow(edges)
plt.axis("off")

plt.subplot(2,3,5)
plt.title("Sobel (Manual)")
plt.imshow(sobel_edges, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()