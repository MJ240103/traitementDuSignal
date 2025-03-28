import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def fourier_transform_rgb(img):
    # séparation des canaux couleur
    r, g, b = cv2.split(img.astype(np.float32))

    # FFT 2D sur chaque canal
    fft_r = np.fft.fft2(r)
    fft_g = np.fft.fft2(g)
    fft_b = np.fft.fft2(b)

    # décalage du spectre pour le centrer
    fft_r_shifted = np.fft.fftshift(fft_r)
    fft_g_shifted = np.fft.fftshift(fft_g)
    fft_b_shifted = np.fft.fftshift(fft_b)

    # calcul du spectre logarithmique
    spectrum_r = np.log1p(np.abs(fft_r_shifted))
    spectrum_g = np.log1p(np.abs(fft_g_shifted))
    spectrum_b = np.log1p(np.abs(fft_b_shifted))

    # combinaison en une seule image (grayscale)
    spectrum = (spectrum_r + spectrum_g + spectrum_b) / 3

    return (fft_r, fft_g, fft_b), (fft_r_shifted, fft_g_shifted, fft_b_shifted), spectrum

def inverse_fourier_transform_rgb(fft_rgb):
    # TF inverse sur chaque canal
    img_r = np.fft.ifft2(fft_rgb[0]).real
    img_g = np.fft.ifft2(fft_rgb[1]).real
    img_b = np.fft.ifft2(fft_rgb[2]).real

    # reconstruction de l'image
    img_reconstructed = np.clip(cv2.merge([img_r, img_g, img_b]), 0, 255).astype(np.uint8)
    
    return img_reconstructed

# charge image
im = Image.open("D:/TPtraitementSignalJolyCoueron/images/images/Nenuphars.jpg")

# conversion en tableau NumPy
img = np.array(im, dtype=np.float32)

# appliquer la FFT sur les 3 canaux
fft_rgb, fft_shifted, spectrum = fourier_transform_rgb(img)

# appliquer la TF inverse pour retrouver l’image couleur
img_reconstructed = inverse_fourier_transform_rgb(fft_rgb)

# normalisation du spectre
spectrum_norm = (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min()) * 255
spectrum_norm = spectrum_norm.astype(np.uint8)

# affichage des résultats
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(im)
plt.title("Image originale")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(spectrum_norm, cmap="gray")
plt.title("Spectre de Fourier (log)")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(img_reconstructed)
plt.title("Image reconstruite (Couleur)")
plt.axis("off")

plt.show()