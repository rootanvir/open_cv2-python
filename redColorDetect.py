import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image = cv2.imread('source/photo.jpg')

# Convert BGR (OpenCV default) to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Define RGB thresholds for red detection
lower_red = np.array([150,0,  0])
upper_red = np.array([255, 80, 80])

# Create mask using RGB thresholds
mask = cv2.inRange(image_rgb, lower_red, upper_red)

# Apply mask to get red parts of the image
result = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

# Display original and detected red side by side
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(result)
plt.title('Red Detection (RGB)')
plt.axis('off')

plt.tight_layout()
plt.show()
