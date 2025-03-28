from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def fourier_transform(img):
    # conversion de l'image en niveaux de gris si nécessaire
    if img.mode != "L":
        img = img.convert("L")

    # conversion en tableau numpy
    arr = np.array(img, dtype=np.float32)

    # transformée de Fourier 2D
    fft = np.fft.fft2(arr)
    # centrage du spectre
    fft_shifted = np.fft.fftshift(fft)
    # echelle logarithmique pour la visualisation
    log_spectrum = np.log1p(np.abs(fft_shifted))

    return fft, fft_shifted, log_spectrum

def inverse_fourier_transform(fft_shifted):
    # décalage inverse pour revenir à l'image de base
    fft_ishifted = np.fft.ifftshift(fft_shifted)
    # TF de Fourier
    img_reconstructed = np.fft.ifft2(fft_ishifted).real
    # conversion 8 bits (valeurs entre 0 et 255)
    return np.clip(img_reconstructed, 0, 255).astype(np.uint8)

#charge image
# charge image dans python
im = Image.open("D:/TPtraitementSignalJolyCoueron/images/images/Nenuphars.jpg")
im2 = Image.open("D:/TPtraitementSignalJolyCoueron/images/images/Bandes_sinus.jpeg")
im3 = Image.open("D:/TPtraitementSignalJolyCoueron/images/images/Bandes_rect.jpeg")
im4 = Image.open("D:/TPtraitementSignalJolyCoueron/images/images/Damier_sinus.jpeg")
im5 = Image.open("D:/TPtraitementSignalJolyCoueron/images/images/Damier_rect.jpeg")
im6 = Image.open("D:/TPtraitementSignalJolyCoueron/imagesRapportTP/3.2quatreBits.png")
im7 = Image.open("D:/TPtraitementSignalJolyCoueron/imagesRapportTP/3.2unBit.jpeg")
im8 = Image.open("D:/TPtraitementSignalJolyCoueron/imagesRapportTP/3.2quatreBits.png")

# appliquer la FFT et récupérer les résultats
fft, fft_shifted, spectrum = fourier_transform(im2)

# appliquer la TF inverse
img_reconstructed = inverse_fourier_transform(fft_shifted)

# normalisation du spectre pour l'affichage
spectrum_norm = (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min()) * 255
spectrum_norm = spectrum_norm.astype(np.uint8)

# affichage des graphes
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(im2, cmap="gray")
plt.title("Image originale")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(spectrum_norm, cmap="gray")
plt.title("Spectre de Fourier (log)")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(img_reconstructed, cmap="gray")
plt.title("Image reconstruite")
plt.axis("off")

plt.tight_layout()
plt.show()
