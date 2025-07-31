import cv2
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('source/photo.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply binary thresholding (threshold value = 127, max value = 255)
_, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)

# Display grayscale and thresholded image side-by-side
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(gray, cmap='gray')
plt.title("Grayscale")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(thresh, cmap='gray')
plt.title("Thresholded (Binary)")
plt.axis("off")

plt.tight_layout()
plt.show()
