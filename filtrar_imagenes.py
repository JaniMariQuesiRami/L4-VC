import os
import cv2
from Anisotropic import anisodiff 
from tqdm import tqdm

#  Ruta de la carpeta con las imágenes de prueba
carpeta_test = 'test/'
carpeta_salida = 'imagenes_filtradas/'

# Crear carpeta de salida si no existe
os.makedirs(carpeta_salida, exist_ok=True)

# Parámetros sugeridos
params = {
    "niter": 50,
    "kappa": 20,
    "gamma": 0.2,
    "step": (1., 1.),
    "option": 1,
    "ploton": False
}

# Recorremos imágenes y aplicamos el filtro
for archivo in tqdm(os.listdir(carpeta_test)):
    if archivo.endswith(('.jpg', '.png')):
        ruta = os.path.join(carpeta_test, archivo)
        img = cv2.imread(ruta)
        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        filtrada = anisodiff(gris, **params)
        salida = os.path.join(carpeta_salida, archivo.replace('.jpg', '_filtrada.png'))
        cv2.imwrite(salida, filtrada)
