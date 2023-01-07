import cv2

#changer le label en fonction du dataset
label = input("Entree le label : ")
cnt_img = 0

#Initialisation de la webcamA
capture = cv2.VideoCapture(0)

while True:
    #capture d'une image du flux de la webcam
    ret,img = capture.read()
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
        #Ecrit l'image dans le fichier
        img_name = "image\{0}\{0}_{1}.png".format(label,cnt_img)
        cv2.imwrite(img_name, img)
        print("{} ecrit!".format(img_name))
        cnt_img += 1


capture.release()
cv2.destroyAllWindows()