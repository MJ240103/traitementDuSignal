import cv2
import numpy as np
import matplotlib.pyplot as plt

def convolution(image, kernel):
    # dimensions image et noyau
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    # calcul des offsets du noyau
    pad_h = kernel_height // 2
    pad_w = kernel_width // 2
    
    # ajouter du padding autour de l'image
    # on ajoute des zéros autour de l'image pour éviter les erreurs de débordement.
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    
    # initialiser la sortie avec une matrice de même taille que l'image originale
    output = np.zeros((image_height, image_width))
    
    # convolution
    for i in range(image_height):
        for j in range(image_width):
            #pour chaque pixel de l'image on extrait une région de même taille que le noyau
            region = padded_image[i:i+kernel_height, j:j+kernel_width]
            # on applique la multiplication élément par élément et on somme le tout
            output[i, j] = np.sum(region * kernel)
    
    return output

# Exemple
image = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]])

kernel = np.array([[0, 1, 0],
                   [1, -4, 1],
                   [0, 1, 0]])

result = convolution(image, kernel)
print(result)