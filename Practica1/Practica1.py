import cv2
import numpy as np


points = []
drawing = False

# Función para manejar los clics del mouse
def draw_circle(event, x, y, flags, param):
    global points, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
        if len(points) > 1:
            cv2.line(img, points[-2], points[-1], (255, 0, 0), 2)
    elif event == cv2.EVENT_LBUTTONUP:
        cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
        if len(points) > 1:
            cv2.line(img, points[-2], points[-1], (255, 0, 0), 2)
        drawing = False

# Función para recortar la imagen 
def recortar_y_guardar():
    if len(points) > 2:  # Asegurarse de que haya al menos 3 puntos
        # Cerrar el camino uniendo el último punto con el primero
        cv2.line(img, points[-1], points[0], (255, 0, 0), 2)

        # Obtener las coordenadas del recorte (punto superior izquierdo y punto inferior derecho)
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x_min, y_min = min(xs), min(ys)
        x_max, y_max = max(xs), max(ys)

        # Realizar el recorte
        recorte = original_img[y_min:y_max, x_min:x_max]

        # Guardar la imagen recortada
        cv2.imwrite('imagen_recortada.png', recorte)
        print("Imagen recortada guardada como 'imagen_recortada.jpg'")

# Cargar la imagen
img = cv2.imread('Captura.png')
original_img = img.copy()

# Crear una ventana y asignar la función del mouse
cv2.namedWindow('Imagen')
cv2.setMouseCallback('Imagen', draw_circle)

while True:
    cv2.imshow('Imagen', img)
    
    # Si se presiona la tecla 's', se guarda la imagen recortada
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        recortar_y_guardar()
        break
    elif key == 27:  # tecla ESC para salir sin guardar
        break

cv2.destroyAllWindows()