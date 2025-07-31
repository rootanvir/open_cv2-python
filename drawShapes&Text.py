import cv2
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('photo.jpg')

# Draw a blue rectangle: (start_x, start_y), (end_x, end_y), color, thickness
cv2.rectangle(image, (50, 50), (250, 250), (255, 0, 0), 3)

# Draw a green circle: center_x, center_y, radius, color, thickness
cv2.circle(image, (150, 150), 50, (0, 255, 0), 3)

# Draw a red line: start_point, end_point, color, thickness
cv2.line(image, (0, 0), (300, 300), (0, 0, 255), 2)

# Add yellow text: text, start_point, font, font_scale, color, thickness
cv2.putText(image, 'OpenCV Demo', (60, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

# Convert to RGB for display
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Show the image
plt.imshow(image_rgb)
plt.title("Shapes and Text")
plt.axis("off")
plt.show()
