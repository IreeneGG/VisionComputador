# Proyecto de Detección y Clasificación de Frutas en Tiempo Real 🔎🍇

Este repositorio contiene los archivos y el código fuente necesarios para el desarrollo de un sistema de visión por computador que detecta y clasifica frutas en tiempo real, evaluando tanto el tipo de fruta como su estado (bueno o malo).
El proyecto se divide en dos etapas principales, cada una de las cuales cuenta con un archivo específico en este repositorio:

## 1. ExtraerCaracteristicasDataset.ipynb 🗂️
Este archivo es un notebook de Google Colab diseñado para la extracción de características de un conjunto de datos de imágenes almacenado en Google Drive.

	python parctica.py

### Uso
1. Abre el archivo en Google Colab.
2. Asegúrate de montar Google Drive para acceder al dataset.
3. Ejecuta las celdas para procesar las imágenes del dataset y guardar las características en un archivo CSV.

## 1. proyecto.py 🗂️
Este archivo contiene el código necesario para la captura de imágenes en tiempo real mediante una cámara conectada al sistema.

### Uso
1. Conecta una cámara compatible al sistema.
2. Ejecuta el archivo proyecto.py.
3. Coloca una fruta frente a la cámara y presiona una tecla para capturar y procesar la imagen.
