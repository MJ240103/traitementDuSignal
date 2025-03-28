from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# charge image dans python
im = Image.open("D:/TPtraitementSignalJolyCoueron/images/images/Nenuphars.jpg")
im2 = Image.open("D:/TPtraitementSignalJolyCoueron/images/images/Bandes_sinus.jpeg")
im3 = Image.open("D:/TPtraitementSignalJolyCoueron/images/images/Bandes_rect.jpeg")
im4 = Image.open("D:/TPtraitementSignalJolyCoueron/images/images/Damier_sinus.jpeg")
im5 = Image.open("D:/TPtraitementSignalJolyCoueron/images/images/Damier_rect.jpeg")
im6 = Image.open("D:/TPtraitementSignalJolyCoueron/imagesRapportTP/3.2quatreBits.png")
im7 = Image.open("D:/TPtraitementSignalJolyCoueron/imagesRapportTP/3.2unBit.png")
im8 = Image.open("D:/TPtraitementSignalJolyCoueron/imagesRapportTP/3.2quatreBits.png")

# affiche informations images, permet de voir si image bien chargé
print(im.format, im.size, im.mode)

# affiche image dans une autre fenêtre
im.show()

# 3.1
# pour 8 bits, on a 256 nuances de gris
def filtreRgb_nb(img):
    # copie de l'image originale pour ne pas la détériorer
    img_bis = img.copy()
    # Parcourir chaque pixel et appliquer la conversion en niveaux de gris
    for j in range(img.height):
        for i in range(img.width):
            r, g, b = img.getpixel((i, j))
            # Conversion en niveaux de gris
            gray = int(0.299 * r + 0.587 * g + 0.114 * b)
            # replace pixel avec nouvelle couleur
            img_bis.putpixel((i, j), (gray, gray, gray))
    
    return img_bis

# affiche image dans une autre fenêtre
filtreRgb_nb(im).save("test.jpg")
filtreRgb_nb(im).show()

#3.2
# pour 4 bits, on a 16 nuances de gris
def filtreNb_4bits(img):
    # copie de l'image origale pour ne pas la détériorer
    img_bis = img.copy()
    # liste des pixels de l'image
    arr = np.array(img_bis)
    # Réduction en 16 niveaux
    arr_4bit = (arr // 16) * 16
    
    # Assurer que les valeurs sont bien dans la plage [0, 255]
    arr_4bit = np.clip(arr_4bit, 0, 255)

    # Convertir en image Pillow et sauvegarder
    img_4bit = Image.fromarray(arr_4bit.astype("uint8"))
    
    return img_4bit

# affiche image dans une autre fenêtre
filtreNb_4bits(filtreRgb_nb(im)).show()

# pour 2 bits, on a 4 nuances de gris
def filtreNb_2bits(img):
    # copie de l'image origale pour ne pas la détériorer
    img_bis = img.copy()
    # liste des pixels de l'image
    arr = np.array(img_bis)
    # Réduction des niveaux de gris à 4 niveaux (0, 85, 170, 255)
    # Division par 64, puis multiplication par 85 pour obtenir 4 niveaux
    arr_2bit = (arr // 64) * 85
    
    # Assurer que les valeurs sont bien dans la plage [0, 255]
    arr_2bit = np.clip(arr_2bit, 0, 255)

    # Convertir en image Pillow et sauvegarder
    img_2bit = Image.fromarray(arr_2bit.astype("uint8"))
    
    return img_2bit

# affiche image dans une autre fenêtre
filtreNb_2bits(filtreRgb_nb(im)).show()

# pour 1 bits, on a 2 nuances de gris
def filtreNb_1bits(img):
    # copie de l'image origale pour ne pas la détériorer
    img_bis = img.copy()
    # liste des pixels de l'image
    arr = np.array(img_bis)
    # Conversion à 1 bit (noir et blanc) en utilisant un seuil de 128
    arr_1bit = np.where(arr >= 128, 255, 0)  # Pixels >= 128 deviennent blancs, sinon noirs

    # Convertir en image Pillow et sauvegarder
    img_1bit = Image.fromarray(arr_1bit.astype("uint8"))
    
    return img_1bit

# affiche image dans une autre fenêtre
filtreNb_1bits(filtreRgb_nb(im)).show()

#3.3
def sous_echantillonnage(img, n):
    # image -> tableau numpy
    arr = np.array(img)

    # dimensions de l'image
    # h : hauteur, w : largeur, c : rgb
    h, w, c = arr.shape
    
    # tableau pour l'image sous-échantillonnée
    new_h = h // n
    new_w = w // n
    arr_resized = np.zeros((new_h, new_w, c), dtype=np.uint8)
    
    # Parcours de l'image en blocs de taille n x n
    for i in range(0, new_h):
        for j in range(0, new_w):
            # Récupérer le premier pixel du bloc (en haut à gauche)
            first_pixel = arr[i * n, j * n]

            # Stocker la couleur du premier pixel dans l'image réduite
            arr_resized[i, j] = first_pixel
    
    # tableau numpy -> image
    img_resized = Image.fromarray(arr_resized)

    return img_resized

# affiche image dans une autre fenêtre
filtreNb_4bits(filtreRgb_nb(im)).save("image_4bit.jpg")
sous_echantillonnage(filtreRgb_nb(im2), 3).show()
sous_echantillonnage(filtreRgb_nb(im2), 10).show()
sous_echantillonnage(filtreRgb_nb(im4), 3).show()
sous_echantillonnage(filtreRgb_nb(im4), 10).show()