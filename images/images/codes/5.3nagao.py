from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

def nagao_filter(image, window_size=5):
    # vérifier que l'image est en niveaux de gris
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ajout d'un padding pour gérer les bords
    pad = window_size // 2
    padded_image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)

    # image de sortie
    filtered = np.zeros_like(image, dtype=np.float32)

    # parcourir chaque pixel de l'image originale
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Extraire la fenêtre 5x5 autour du pixel
            window = padded_image[i:i+window_size, j:j+window_size]

            # Définit les 5 sous-régions de Nagao
            subregions = [
                window[0:3, 0:3],  # Haut-gauche
                window[0:3, 2:5],  # Haut-droite
                window[2:5, 0:3],  # Bas-gauche
                window[2:5, 2:5],  # Bas-droite
                window[1:4, 1:4]   # Centre
            ]

            # Calcule variance de chaque sous-région
            variances = [np.var(region) for region in subregions]

            # région avec la plus petite variance
            min_var_index = np.argmin(variances)

            # remplace le pixel par la moyenne de cette région
            filtered[i, j] = np.mean(subregions[min_var_index])

    return filtered.astype(np.uint8)

# charger image en niveaux de gris
image = cv2.imread("D:/TPtraitementSignalJolyCoueron/images/images/Nenuphars.jpg", cv2.IMREAD_GRAYSCALE)

# filtre de Nagao
filtered_nagao = nagao_filter(image)

# résultats
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(image, cmap='gray')
axes[0].set_title("Image Originale")
axes[0].axis("off")

axes[1].imshow(filtered_nagao, cmap='gray')
axes[1].set_title("Image Filtrée (Nagao)")
axes[1].axis("off")

plt.show()