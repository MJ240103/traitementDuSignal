import cv2
import numpy as np
import matplotlib.pyplot as plt

# charger image en niveaux de gris
image = cv2.imread("D:/TPtraitementSignalJolyCoueron/images/images/Nenuphars.jpg", cv2.IMREAD_GRAYSCALE)

# Définition filtres
gradient_x = np.array([[1, 0, -1],
                       [1, 0, -1],
                       [1, 0, -1]], dtype=np.float32)

laplacien_cross = np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]], dtype=np.float32)

# filtres
gradient_x_filtered = cv2.filter2D(image, -1, gradient_x)
laplacien_filtered = cv2.filter2D(image, -1, laplacien_cross)

# résultats
plt.figure(figsize=(10,5))
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Image originale')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(gradient_x_filtered, cmap='gray')
plt.title('Gradient selon x')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(laplacien_filtered, cmap='gray')
plt.title('Laplacien en croix')
plt.axis('off')

plt.show()