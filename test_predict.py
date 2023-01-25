from keras.models import load_model
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator


#Initialisation de la webcamA
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


# coordonnées du rectangle a changé en fonction du label
x1 = 0 + 500
y1 = 0 + 250
x2 = 1280 - 500
y2 = 720 - 200

while True:
    #capture d'une image du flux de la webcam
    ret,img = capture.read()
    if not ret:
        print("Erreur lors de la lecture de img")
        break

    # Trace le rectangle sur l'image
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    #Affichage de l'image
    cv2.imshow("Image", img)

    key = cv2.waitKey(1)
    # ESC 
    if key%256 == 27:
        print("ESC, fermeture...")
        break
    # ESPACE
    elif key%256 == 32:
        #Ecrit le roi dans le fichier
        new_image = img[y1+2:y2-1, x1+2:x2-1]
        # Load the model from the H5 file
        loaded_model = load_model("test.h5")

        # Load the new image you want to predict
        #new_image = cv2.imread("images/validation/I/I_1.png")

        # Resize the image to the same size as the training images
        new_image = cv2.resize(new_image, (50, 50))

        # Convert the image to a numpy array and normalize the pixel values
        new_image = np.array(new_image) / 255.0

        # Add a batch dimension to the image
        new_image = np.expand_dims(new_image, axis=0)

        # Use the predict method to get the model's predictions
        predictions = loaded_model.predict(new_image)

        # Get the index of the class with the highest probability
        class_index = np.argmax(predictions[0])

        print(predictions)

        print("Predicted class index:", class_index)

capture.release()
cv2.destroyAllWindows()