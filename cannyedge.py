import cv2
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('photo.jpg')

# Convert to grayscale (required for Canny)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Perform Canny edge detection (threshold1, threshold2)
edges = cv2.Canny(blurred, 50, 150)

# Display original and edges side-by-side
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title("Canny Edges")
plt.axis("off")

plt.tight_layout()
plt.show()
