import cv2
import numpy as np
from collections import Counter

def process_image(image_path):
  
    image = cv2.imread(image_path)
    # Redimensionar
    image = cv2.resize(image, (600, 400))
    
    # Preprocesamiento
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Crear copia de imagen original
    output = image.copy()
    cv2.drawContours(output, contours, -1, (0, 255, 0), 2)  # Dibujar contornos en verde
    
    return image, edges, output, contours

def get_color_ranges(image, contours):
    # Crear una máscara vacía
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Dibujar los contornos en la máscara
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

    # Aplicar la máscara a la imagen original
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    # Convertir la imagen en BGR a HSV
    hsv_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)

    # Extraer el canal H
    h_channel = hsv_image[:, :, 0]
    
    # Contar los colores en el canal H
    hist = cv2.calcHist([h_channel], [0], mask, [180], [0, 180])
    
    # Encontrar los índices del histograma más altos
    dominant_hues = np.argsort(hist.flatten())[-5:]  # Obtener los 5 más comunes
    
    # Crear rangos de colores
    ranges = [(h, h + 10) for h in dominant_hues]  # Definir un rango para cada matiz
    return ranges

def calculate_simple_rugosity(image):
    # Convertir a gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calcular la varianza
    rugosity = np.var(gray)  # Varianza como medida de rugosidad
    
    return rugosity

# Uso
image_path = 'limonM2.jpg'  
original_image, edges_image, contours_image, contours = process_image(image_path)

# Obtener los rangos de color predominante
color_ranges = get_color_ranges(original_image, contours)

# Calcular la rugosidad
rugosity = calculate_simple_rugosity(original_image)

# Mostrar en ventanas diferentes
cv2.imshow('Imagen Original', original_image)
cv2.imshow('Bordes Detectados', edges_image)
cv2.imshow('Contornos Detectados', contours_image)

# Imprimir los rangos de colores
print("Rangos de colores predominantes (H):")
for h_range in color_ranges:
    print(f"Rango: {h_range}")

# Imprimir la rugosidad calculada
print("Rugosidad calculada:", rugosity)

cv2.waitKey(0)
cv2.destroyAllWindows()
