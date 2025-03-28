import cv2
import numpy as np
import matplotlib.pyplot as plt

# charger l'image en niveaux de gris
image = cv2.imread("D:/TPtraitementSignalJolyCoueron/images/images/Nenuphars.jpg", cv2.IMREAD_GRAYSCALE)

# vérifie si l'image est bien chargée
if image is None:
    raise ValueError("L'image n'a pas pu être chargée. Vérifiez le chemin du fichier.")

# définition du noyau de filtrage
kernel = np.ones((3, 3), np.float32) / 9

# appliquer la convolution avec le noyau
filtered_image = cv2.filter2D(image, -1, kernel)

# résultats
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(image, cmap='gray')
axes[0].set_title("Image Originale")
axes[0].axis("off")

axes[1].imshow(filtered_image, cmap='gray')
axes[1].set_title("Image Filtrée (Passe-bas) matrice")
axes[1].axis("off")

plt.show()