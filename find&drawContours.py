import cv2
import matplotlib.pyplot as plt

# Load image and convert to grayscale
image = cv2.imread('source/photo.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Thresholding to get binary image
_, thresh = cv2.threshold(gray,120, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on a copy of original image
image_contours = image.copy()
cv2.drawContours(image_contours, contours, -1, (0, 255, 0), 2)

# Convert to RGB for matplotlib
image_rgb = cv2.cvtColor(image_contours, cv2.COLOR_BGR2RGB)

# Show result
plt.imshow(image_rgb)
plt.title('Contours Detected')
plt.axis('off')
plt.show()
