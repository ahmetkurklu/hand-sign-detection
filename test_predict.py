from keras.models import load_model
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from cvzone import HandTrackingModule


#Initialisation de la webcamA
capture = cv2.VideoCapture(0)
detector = HandTrackingModule.HandDetector()
loaded_model = load_model("model_save/test_avec_crop_mediapipe.h5")

while True:
    #capture d'une image du flux de la webcam
    ret,img = capture.read()
    ret,img = capture.read()
    img_copy = img.copy()
    hands, img = detector.findHands(img)
    
    if not ret:
        print("Erreur lors de la lecture de img")
        break

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
        bbox_value = hands[0].get('bbox')
        new_image = img_copy[bbox_value[1]:bbox_value[1] + bbox_value[3], bbox_value[0]:bbox_value[0] + bbox_value[2]]
        # Load the model from the H5 file
        

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

    