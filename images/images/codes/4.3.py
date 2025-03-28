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

def transformation_polynomiale(img, alpha):
    # image -> tableau numpy (float pour éviter les pertes)
    arr = np.array(img, dtype=np.float32)
    print(arr)
    
    # transformation polynomiale
    arr_transfo = ((arr / 256) ** alpha) * 256
    # assurer que les valeurs restent entre 0 et 255
    arr_transfo = np.clip(arr_transfo, 0, 255)
    # reconvertir en entier 8 bits
    arr_transfo = arr_transfo.astype(np.uint8)
    
    img_transfo = Image.fromarray(arr_transfo)
    
    return img_transfo

def afficher_spectre(img, title="Spectre de l'image"):
    arr = np.array(img)
    plt.figure(figsize=(6, 4))
    plt.hist(arr.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.7)
    plt.title(title)
    plt.xlabel("Niveau d'intensité")
    plt.ylabel("Nombre de pixels")
    plt.show()

# Appliquer la transformation avec différents α
alpha1 = 2.0  # Assombrissement
alpha2 = 0.5  # Éclaircissement

img_sombre = transformation_polynomiale(im8, alpha1)
img_claire = transformation_polynomiale(im8, alpha2)

# Afficher les images
plt.figure(figsize=(15, 5))

plt.subplot(1, 4, 1)
plt.imshow(im, cmap="gray")
plt.title("Image originale")
plt.subplot(1, 4, 2)
plt.imshow(im8, cmap="gray")
plt.title("Image couleur")

plt.subplot(1, 4, 3)
plt.imshow(img_sombre, cmap="gray")
plt.title(f"Assombri (α = {alpha1})")

plt.subplot(1, 4, 4)
plt.imshow(img_claire, cmap="gray")
plt.title(f"Éclairci (α = {alpha2})")

plt.show()

# Afficher les spectres
afficher_spectre(im8, "Spectre original")
afficher_spectre(im, "Spectre couleur")
afficher_spectre(img_sombre, f"Spectre assombri (α={alpha1})")
afficher_spectre(img_claire, f"Spectre éclairci (α={alpha2})")