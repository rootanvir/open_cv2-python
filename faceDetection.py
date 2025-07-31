import cv2
import matplotlib.pyplot as plt

# Load the Haar cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Read your local image
image = cv2.imread('source/elon.jpg')  # Make sure this filename is correct
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3)
print("Faces detected:", len(faces))

# Draw rectangles around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Convert BGR to RGB for displaying in matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Show result
plt.imshow(image_rgb)
plt.title('Bill Gates - Face Detection')
plt.axis('off')
plt.show()
