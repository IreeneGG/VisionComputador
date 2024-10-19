import cv2
import numpy as np

# Inicializar la captura de la cámara
captura = cv2.VideoCapture(0)

# Definir el rango de color de piel en YCrCb
bajo_piel = np.array([0, 132, 77], dtype=np.uint8)   # Límite inferior 
alto_piel = np.array([200, 173, 127], dtype=np.uint8)  # Límite superior 

while True:
    ret, frame = captura.read()
   
    # Invertir la cámara
    flipped_frame = cv2.flip(frame, 1)

    # Convertir a escala de grises
    gris = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2GRAY)

    # --- FILTRO DE PASO BAJO ---
    # Suavizado usando un desenfoque gaussiano
    imagen_suavizada = cv2.GaussianBlur(gris, (5, 5), 0)

    # --- FILTRO CANNY ---
    # Aplicar el detector de bordes
    bordes = cv2.Canny(imagen_suavizada, 100, 200)

    # --- FILTRO THRESHOLD ---
    # Umbralización para obtener una imagen binaria
    _, imagen_binaria = cv2.threshold(bordes, 128, 255, cv2.THRESH_BINARY)

    # Convertir el cuadro a espacio de color YCrCb
    ycrcb = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2YCrCb)

    # Crear una máscara para el color de la piel en YCrCb
    mascara_piel = cv2.inRange(ycrcb, bajo_piel, alto_piel)

    # --- Aplicar operaciones morfológicas para reducir el ruido ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Eliminar el ruido pequeño
    mascara_piel = cv2.morphologyEx(mascara_piel, cv2.MORPH_OPEN, kernel)

    # Aplicar una operación de cierre para unir regiones de la piel
    mascara_piel = cv2.morphologyEx(mascara_piel, cv2.MORPH_CLOSE, kernel)

    # Crear una imagen de salida donde el fondo será negro
    salida = np.zeros_like(flipped_frame)

    # Aplicar la máscara a la imagen original
    salida[mascara_piel != 0] = flipped_frame[mascara_piel != 0]

    # Crear una imagen para mostrar solo las líneas verdes
    lineas_verdes = np.zeros_like(flipped_frame)

    # Encontrar los contornos en la imagen binaria
    contornos, _ = cv2.findContours(imagen_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- FILTRO CONVEXHULL ---
    # Dibujar la envolvente convexa sobre la imagen de líneas verdes
    for contorno in contornos:
        hull = cv2.convexHull(contorno)
        cv2.drawContours(lineas_verdes, [hull], -1, (0, 255, 0), thickness=2)  # Dibujar en verde

    # Mostrar las diferentes ventanas
    cv2.imshow('Imagen Suavizada (GaussianBlur)', imagen_suavizada)
    cv2.imshow('Bordes (Canny)', imagen_binaria)
    cv2.imshow('Líneas Verdes (Convex Hull)', lineas_verdes)
    cv2.imshow('Resultado final', salida)

    # Capturar la tecla presionada
    key = cv2.waitKey(1) & 0xFF
    
    # Guardar una captura si se presiona la tecla 's'
    if key == ord('s'):
        cv2.imwrite('practica2.png', salida)
        print("Imagen guardada como practica2.png")
    
    # Salir del bucle si se presiona la tecla 'q' o 'ESC'
    elif key == ord('q') or key == 27:
        break

# Liberar los recursos
captura.release()
cv2.destroyAllWindows()
