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

def afficher_spectre(img):
    # iimage -> tableau numpy
    arr = np.array(img)
    
    # Séparer les canaux de couleur (R, V, B) grâce au slicing
    r, v, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    
    # Calcul du niveau de gris (moyenne des canaux)
    niveaux_gris = (0.299 * r + 0.587 * v + 0.114 * b).astype(np.uint8)
    
    # Création de la figure avec 4 histogrammes
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    
    # Histogramme Rouge
    axs[0].hist(r.ravel(), bins=256, color='red', alpha=0.7)
    axs[0].set_title("Spectre Rouge")
    
    # Histogramme Vert
    axs[1].hist(v.ravel(), bins=256, color='green', alpha=0.7)
    axs[1].set_title("Spectre Vert")
    
    # Histogramme Bleu
    axs[2].hist(b.ravel(), bins=256, color='blue', alpha=0.7)
    axs[2].set_title("Spectre Bleu")
    
    # Histogramme Niveaux de Gris
    axs[3].hist(niveaux_gris.ravel(), bins=256, color='black', alpha=0.7)
    axs[3].set_title("Spectre Niveaux de Gris")
    
    # Afficher les histogrammes
    plt.show()

# Afficher le spectre de l'image
afficher_spectre(im) 
afficher_spectre(im6)
afficher_spectre(im7)