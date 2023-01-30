import cv2
from cvzone import HandTrackingModule

#changer le label en fonction du dataset
label = input("Entree le label : ")
cnt_img = 0
detector = HandTrackingModule.HandDetector()

#Initialisation de la webcam
capture = cv2.VideoCapture(0)

while True:
    #capture d'une image du flux de la webcam
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

        bbox_value = hands[0].get('bbox')

        #Ecrit le roi dans le fichier
        roi = img_copy[bbox_value[1]:bbox_value[1] + bbox_value[3], bbox_value[0]:bbox_value[0] + bbox_value[2]]

        img_name = "images_rph/{0}/{0}_{1}r.png".format(label,cnt_img)
        cv2.imwrite(img_name, roi)
        print("{} ecrit!".format(img_name))
        cnt_img += 1


capture.release()
cv2.destroyAllWindows()       
    