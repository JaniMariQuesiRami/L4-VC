import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from tqdm import tqdm

# Cargar modelo entrenado
modelo = load_model('mejor_unet.h5')

# Tamaño de ventana (debe coincidir con entrenamiento)
k = 32
stride = k // 2

# Carpetas
carpeta_entrada = 'BSDS500/data/images/test/'
carpeta_salida = 'reconstruidas_por_unet/'
os.makedirs(carpeta_salida, exist_ok=True)

# Procesar cada imagen de test
for nombre_archivo in tqdm(os.listdir(carpeta_entrada)):
    if not nombre_archivo.endswith('.jpg'):
        continue

    # Cargar imagen en escala de grises
    ruta_imagen = os.path.join(carpeta_entrada, nombre_archivo)
    img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    alto, ancho = img.shape

    # Inicializar matrices de reconstrucción
    reconstruida = np.zeros((alto, ancho), dtype=np.float32)
    conteo = np.zeros((alto, ancho), dtype=np.float32)

    # Deslizar ventana con stride
    for y in range(0, alto - k + 1, stride):
        for x in range(0, ancho - k + 1, stride):
            ventana = img[y:y+k, x:x+k].astype(np.float32) / 255.0
            ventana = np.expand_dims(ventana, axis=(0, -1))  # (1, k, k, 1)

            # Inferencia
            salida = modelo.predict(ventana, verbose=0)[0, ..., 0]

            reconstruida[y:y+k, x:x+k] += salida
            conteo[y:y+k, x:x+k] += 1

    # Promediar valores traslapados
    final = (reconstruida / np.maximum(conteo, 1e-5)) * 255.0
    final = np.clip(final, 0, 255).astype(np.uint8)

    # Guardar imagen reconstruida
    nombre_salida = os.path.splitext(nombre_archivo)[0] + '_unet.png'
    cv2.imwrite(os.path.join(carpeta_salida, nombre_salida), final)

print("✅ Todas las imágenes fueron filtradas por la U-Net y guardadas en:", carpeta_salida)
