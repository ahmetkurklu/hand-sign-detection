from keras.models import load_model
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from cvzone import HandTrackingModule


#Initialisation de la webcamA
capture = cv2.VideoCapture(0)
detector = HandTrackingModule.HandDetector()
loaded_model = load_model("model_save/model_avec_gauss.h5")

classData = {
    0 : "A",
    1 : "B",
    2 : "C",
    3 : "G",
    4 : "H",
    5 : "I",
    6 : "L",
    7 : 'R',
    8 : 'V',
    9 : 'W'
}
while True:
    #capture d'une image du flux de la webcam
    # ret,img = capture.read()
    ret,img = capture.read()
    img_copy = img.copy()
    hands, img = detector.findHands(img)
    
    if not ret:
        print("Erreur lors de la lecture de img")
        break

    

    key = cv2.waitKey(1)
    # ESC 
    if key%256 == 27:
        print("ESC, fermeture...")
        break
    # ESPACE
    # elif key%256 == 32:
    if hands != []:
        #Ecrit le roi dans le fichier
        bbox_value = hands[0].get('bbox')
        new_image = img_copy[bbox_value[1]:bbox_value[1] + bbox_value[3], bbox_value[0]:bbox_value[0] + bbox_value[2]]

        new_image = cv2.GaussianBlur(new_image, (5, 5), 0)
        
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

        # print(predictions)
        results = predictions[0].tolist()
        print(results)
        print("Predicted class index:", class_index,classData[class_index])
        cv2.putText(img, f"signe : {classData[class_index]} [{(results[class_index]):.2f}%]", (30, 30), cv2.FONT_HERSHEY_PLAIN,2, (255, 0, 0), 2)
        
    #Affichage de l'image
    cv2.imshow("Image", img)

capture.release()
cv2.destroyAllWindows()

    