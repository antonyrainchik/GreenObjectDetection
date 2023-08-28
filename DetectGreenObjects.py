import cv2
import numpy as np

# Load the image
image = cv2.imread('path_to_image.jpg')

# Convert the image to the HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the range for green color
lower_green = np.array([35, 50, 50])
upper_green = np.array([85, 255, 255])

# Create a binary mask where green pixels are white and all other pixels are black
mask = cv2.inRange(hsv, lower_green, upper_green)

# Optional: Apply morphological operations to reduce small noise
kernel = np.ones((5,5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Find the contours in the mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Count the number of contours which correspond to the number of green objects
num_green_objects = len(contours)

print(f'The number of green objects in the image is: {num_green_objects}')

# Optional: Visualize the result
for contour in contours:
    cv2.drawContours(image, [contour], -1, (0,255,0), 2)

cv2.imshow('Green Objects', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
