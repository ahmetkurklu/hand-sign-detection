import cv2

# Read the image
image = cv2.imread("image/B/B_3.png")


# Load the Haar cascade classifier for hands
hand_cascade = cv2.CascadeClassifier('hand.xml')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_,thresh = cv2.threshold(gray,120,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# Detect hands in the image
hands = hand_cascade.detectMultiScale(thresh, 1.1, 5)

# Draw a bounding box around each hand and extract the ROI
for (x, y, w, h) in hands:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    roi = image[y:y + h, x:x + w]
    cv2.imshow('ROI', roi)
    cv2.waitKey(0)

# Display the original image
cv2.imshow('Original Image', image)

# Display the original image
cv2.imshow('Thresh Image', thresh)

# Wait until the user hits a key
cv2.waitKey(0)

# Destroy all windows
cv2.destroyAllWindows()