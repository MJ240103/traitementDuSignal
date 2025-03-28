import cv2
import numpy as np
import matplotlib.pyplot as plt

def low_pass_filter_fourier(image, radius=30):
    # conversion en niveaux de gris si nécessaire
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # dimensions de l'image
    rows, cols = image.shape
    center_x, center_y = cols // 2, rows // 2
    
    # transformée de Fourier
    dft = np.fft.fft2(image)
    # décalage du spectre pour centrer les basses fréquences
    dft_shifted = np.fft.fftshift(dft)

    # création du masque passe-bas
    mask = np.zeros((rows, cols), np.uint8)
    # cercle centré pour garder les basses fréquences
    cv2.circle(mask, (center_x, center_y), radius, 1, -1)

    # application du masque
    filtered_dft = dft_shifted * mask

    # retour dans le domaine spatial
    dft_inverse = np.fft.ifftshift(filtered_dft)  # Recentrer
    image_filtered = np.fft.ifft2(dft_inverse)  # TF inverse
    image_filtered = np.abs(image_filtered)  # Module complexe

    return np.uint8(image_filtered)

# Exécuter le filtre
image_path = "D:/TPtraitementSignalJolyCoueron/images/images/Nenuphars.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is not None:
    filtered_image = low_pass_filter_fourier(image, radius=30)
    filtered_image2 = low_pass_filter_fourier(image, radius=60)
    filtered_image3 = low_pass_filter_fourier(image, radius=200)

    # Affichage des résultats
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.title("Image originale")
    plt.imshow(image, cmap="gray")

    plt.subplot(2, 2, 2)
    plt.title("Filtrage Passe-Bas (Fourier) rayon_masque=30")
    plt.imshow(filtered_image, cmap="gray")
    
    plt.subplot(2, 2, 3)
    plt.title("Filtrage Passe-Bas (Fourier) rayon_masque=60")
    plt.imshow(filtered_image2, cmap="gray")
    
    plt.subplot(2, 2, 4)
    plt.title("Filtrage Passe-Bas (Fourier) rayon_masque=200")
    plt.imshow(filtered_image3, cmap="gray")

    plt.show()
else:
    print("Erreur : Impossible de charger l'image.")