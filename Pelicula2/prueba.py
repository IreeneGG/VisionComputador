import cv2
import numpy as np

# Función para aplicar el algoritmo GrabCut a cada frame del video
def procesar_frame(frame, fondo):
    # Convertimos el frame a espacio de color HSV para detectar el color verde
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Visualizar la máscara inicial (color verde detectado)
    cv2.imshow('Máscara Inicial (Color Verde Detectado)', mask)

    # Aplicar el filtro Gaussiano para suavizar la máscara
    mask_suavizada = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # Visualizar la máscara suavizada después del filtro Gaussiano
    cv2.imshow('Máscara Suavizada (Filtro Gaussiano)', mask_suavizada)

    # Usar Canny para detección de bordes en la máscara suavizada
    edges = cv2.Canny(mask_suavizada, 170, 200)
    
    # Visualizar la detección de bordes con Canny
    cv2.imshow('Bordes Detectados (Canny)', edges)

    # Crear una máscara inicial para GrabCut
    grabcut_mask = np.where((edges != 0), 1, 0).astype('uint8')

    # Definir un rectángulo alrededor del área que queremos analizar con GrabCut (usamos todo el frame)
    rect = (0, 0, frame.shape[1], frame.shape[0])

    # Crear las matrices que GrabCut necesita para su ejecución
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Ejecutar el algoritmo GrabCut
    cv2.grabCut(frame, grabcut_mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)

    # Convertir la máscara resultante en una binaria (0 o 1) para aplicarla a la imagen
    mask_final = np.where((grabcut_mask == 2) | (grabcut_mask == 3), 1, 0).astype('uint8')

    # Asegurarse de que la máscara tenga tres canales para aplicarla correctamente al frame
    mask_final_3ch = cv2.merge([mask_final, mask_final, mask_final])

    # Aplicar la máscara al frame para extraer el primer plano
    foreground = cv2.bitwise_and(frame, frame, mask=mask_final)

    # Redimensionar la imagen de fondo para que coincida con el tamaño del frame
    fondo_resized = cv2.resize(fondo, (frame.shape[1], frame.shape[0]))

    # Crear la imagen final combinando el primer plano con el nuevo fondo
    background_mask = cv2.bitwise_not(mask_final)
    background = cv2.bitwise_and(fondo_resized, fondo_resized, mask=background_mask)
    resultado = cv2.add(foreground, background)

    return resultado

# Abrir el video y el fondo de imagen
video_path = 'video.avi'
fondo = cv2.imread('fondo.jpg')
cap = cv2.VideoCapture(video_path)

# Obtener las propiedades del video para crear el archivo de salida
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output = cv2.VideoWriter('video_sin_fondo.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

# Procesar cada frame del video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Procesar el frame y obtener el resultado con el fondo reemplazado
    frame_procesado = procesar_frame(frame, fondo)

    # Guardar el frame procesado en el archivo de salida
    output.write(frame_procesado)

    # Mostrar el frame procesado (opcional)
    cv2.imshow('Frame Procesado', frame_procesado)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
output.release()
cv2.destroyAllWindows()
