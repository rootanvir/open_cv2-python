import cv2
import matplotlib.pyplot as plt

image = cv2.imread('img.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

plt.imshow(image)
plt.show()
