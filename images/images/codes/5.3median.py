import cv2
import numpy as np
import matplotlib.pyplot as plt

# charger l'image en niveaux de gris
image = cv2.imread("D:/TPtraitementSignalJolyCoueron/images/images/Nenuphars.jpg", cv2.IMREAD_GRAYSCALE)

# Appliquer un filtrage médian avec un noyau de taille 3x3
filtered_median = cv2.medianBlur(image, 3)

# Affichage des résultats
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(image, cmap='gray')
axes[0].set_title("Image Originale")
axes[0].axis("off")

axes[1].imshow(filtered_median, cmap='gray')
axes[1].set_title("Image Filtrée (Médian)")
axes[1].axis("off")

plt.show()