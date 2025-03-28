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
im7 = Image.open("D:/TPtraitementSignalJolyCoueron/imagesRapportTP/3.2unBit.jpeg")
im8 = Image.open("D:/TPtraitementSignalJolyCoueron/imagesRapportTP/3.2quatreBits.png")

def negatif_image(img):
    # Si image noir-blanc
    if img.mode == "L":
        arr = np.array(img)
        # Inverser les niveaux de gris
        arr_negatif = 255 - arr
        # Reconstruire image en niveaux de gris
        img_negatif = Image.fromarray(arr_negatif.astype("uint8"), mode="L")
    # Si image couleur
    elif img.mode == "RGB":
        arr = np.array(img)
        # Inverser les couleurs RVB
        arr_negatif = 255 - arr
        # Reconstruire image couleur
        img_negatif = Image.fromarray(arr_negatif.astype("uint8"))
    else:
        raise ValueError("Format d'image non supporté (seulement RGB ou L)")

    return img_negatif

# Afficher l'image négative
negatif_image(im).show()
negatif_image(im2).show()
negatif_image(im7).show()