import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image and convert to grayscale
image = cv2.imread('source/photo.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply binary thresholding
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Define a 5x5 rectangular kernel
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# Dilate the image (expand white regions)
dilated = cv2.dilate(thresh, kernel, iterations=1)

# Erode the image (shrink white regions)
eroded = cv2.erode(thresh, kernel, iterations=1)

# Display all images side by side
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.imshow(thresh, cmap='gray')
plt.title('Original Threshold')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(dilated, cmap='gray')
plt.title('Dilated')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(eroded, cmap='gray')
plt.title('Eroded')
plt.axis('off')

plt.tight_layout()
plt.show()
