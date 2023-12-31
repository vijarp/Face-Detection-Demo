import cv2 as cv
import os
import sys
# Read image from your local file system
original_image = cv.imread('faces1.png')

# Convert color image to grayscale for Viola-Jones
grayscale_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)


# Load the classifier and create a cascade object for face detection
parent_directory = os.path.dirname(sys.executable)
path1 = os.path.join(parent_directory,r"cv2\data\haarcascade_frontalface_alt.xml")

face_cascade = cv.CascadeClassifier(path1)


detected_faces = face_cascade.detectMultiScale(grayscale_image)


for (column, row, width, height) in detected_faces:
    cv.rectangle(
        original_image,
        (column, row),
        (column + width, row + height),
        (0, 255, 0),
        2
    )


cv.imshow('Image', original_image)
cv.waitKey(0)
cv.destroyAllWindows()