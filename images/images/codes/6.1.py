import cv2
import numpy as np
import matplotlib.pyplot as plt

# images en niveaux de gris
image1 = cv2.imread("D:/TPtraitementSignalJolyCoueron/images/images/Nenuphars1.jpg", cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread("D:/TPtraitementSignalJolyCoueron/images/images/Nenuphars2.jpg", cv2.IMREAD_GRAYSCALE)

# vérifier que les images ont la même taille
if image1.shape != image2.shape:
    raise ValueError("Les images doivent avoir la même taille pour l'intercorrélation.")

# Calcul de la TF des deux images
f1 = np.fft.fft2(image1)
f2 = np.fft.fft2(image2)

# Calcul intercorrélation croisée via le produit conjugué
cross_power_spectrum = (f1 * np.conj(f2)) / np.abs(f1 * np.conj(f2))
cross_correlation = np.fft.ifft2(cross_power_spectrum)
cross_correlation = np.abs(np.fft.fftshift(cross_correlation))

# cherche pic de corrélation pour déterminer le décalage optimal
dy, dx = np.unravel_index(np.argmax(cross_correlation), cross_correlation.shape)
dy -= image1.shape[0] // 2
dx -= image1.shape[1] // 2

# Décaler image2 pour l'aligner avec image1
image2_aligned = np.roll(image2, shift=(-dy, -dx), axis=(0, 1))

# résultats
plt.figure(figsize=(15,5))
plt.subplot(1, 3, 1)
plt.imshow(image1, cmap='gray')
plt.title('Image 1')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(image2, cmap='gray')
plt.title('Image 2 (originale)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(image2_aligned, cmap='gray')
plt.title(f'Image 2 alignée (dx={dx}, dy={dy})')
plt.axis('off')

plt.show()