import os
import cv2
import numpy as np
from tqdm import tqdm
import random

# Carpetas de entrada de imágenes originales y filtradas
path_originales = 'test/'
path_filtradas = 'imagenes_filtradas/'
ventana_k = 32
num_muestras = 500000

# Inicializamos listas
xi_list = []
yi_list = []

# Listado de archivos válidos
archivos = [f for f in os.listdir(path_originales) if f.endswith('.jpg')]

# Generar ventanas
for archivo in tqdm(archivos):
    # Leer imágenes original y filtrada
    img_ori = cv2.imread(os.path.join(path_originales, archivo), cv2.IMREAD_GRAYSCALE)
    img_fil = cv2.imread(os.path.join(path_filtradas, archivo.replace('.jpg', '_filtrada.png')), cv2.IMREAD_GRAYSCALE)

    alto, ancho = img_ori.shape

    # Cuántas ventanas tomamos de esta imagen
    ventanas_por_imagen = num_muestras // len(archivos)

    for _ in range(ventanas_por_imagen):
        x = random.randint(0, ancho - ventana_k)
        y = random.randint(0, alto - ventana_k)

        xi = img_ori[y:y+ventana_k, x:x+ventana_k]
        yi = img_fil[y:y+ventana_k, x:x+ventana_k]

        xi_list.append(xi)
        yi_list.append(yi)

# Convertir a arrays numpy
X = np.array(xi_list).astype(np.uint8)
Y = np.array(yi_list).astype(np.uint8)

# Añadir dimensión para canales (necesario para U-Net)
X = np.expand_dims(X, axis=-1)  # shape: (N, k, k, 1)
Y = np.expand_dims(Y, axis=-1)

# Dividir en sets (80% train, 10% val, 10% test)
total = X.shape[0]
i_train = int(0.8 * total)
i_val = int(0.9 * total)

X_train, Y_train = X[:i_train], Y[:i_train]
X_val, Y_val = X[i_train:i_val], Y[i_train:i_val]
X_test, Y_test = X[i_val:], Y[i_val:]

# Guardar a disco
np.savez_compressed('ventanas_dataset.npz',
                    X_train=X_train, Y_train=Y_train,
                    X_val=X_val, Y_val=Y_val,
                    X_test=X_test, Y_test=Y_test)

print("✅ Dataset de ventanas generado y guardado en 'ventanas_dataset.npz'")
