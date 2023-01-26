import cv2

#changer le label en fonction du dataset
label = input("Entree le label : ")
cnt_img = 0

#Initialisation de la webcamA
capture = cv2.VideoCapture(0)
x=720
y=640
capture.set(cv2.CAP_PROP_FRAME_WIDTH, x)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, y)


# coordonnées du rectangle a changé en fonction du label
x1 = 0
y1 = 0
x2 = x - 500
y2 = y - 400

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
        roi = img[y1+2:y2-1, x1+2:x2-1]
        img_name = ".\images_rph\{0}\{0}_{1}r.png".format(label,cnt_img)
        cv2.imwrite(img_name, roi)
        print("{} ecrit!".format(img_name))
        cnt_img += 1


capture.release()
cv2.destroyAllWindows()