import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, optimizers, callbacks

# ✅ Configurar GPU
tf.debugging.set_log_device_placement(True)  # Mostrar en consola qué se ejecuta en GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"{len(gpus)} GPU(s) habilitada(s): {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        print("Error al configurar crecimiento de memoria:", e)
else:
    print("❌ No se detectaron GPUs. Usando CPU.")

# ✅ Cargar el dataset
datos = np.load('ventanas_dataset.npz')
X_train, Y_train = datos['X_train'], datos['Y_train']
X_val, Y_val = datos['X_val'], datos['Y_val']

k = X_train.shape[1]

# ✅ Definir modelo U-Net
def unet_model(input_size=(k, k, 1)):
    inputs = layers.Input(input_size)

    # Encoder
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(c3)

    # Decoder
    u2 = layers.UpSampling2D((2, 2))(c3)
    u2 = layers.concatenate([u2, c2])
    c4 = layers.Conv2D(64, 3, activation='relu', padding='same')(u2)
    c4 = layers.Conv2D(64, 3, activation='relu', padding='same')(c4)

    u1 = layers.UpSampling2D((2, 2))(c4)
    u1 = layers.concatenate([u1, c1])
    c5 = layers.Conv2D(32, 3, activation='relu', padding='same')(u1)
    c5 = layers.Conv2D(32, 3, activation='relu', padding='same')(c5)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c5)

    return models.Model(inputs=[inputs], outputs=[outputs])

# ✅ Compilar y entrenar
model = unet_model()
model.compile(optimizer=optimizers.Adam(1e-4), loss='mse', metrics=['mae'])
model.summary()

checkpoint = callbacks.ModelCheckpoint('mejor_unet_gpu.h5', monitor='val_loss',
                                       save_best_only=True, verbose=1)

# Normalizamos los valores de los píxeles a [0, 1]
X_train = X_train.astype('float32') / 255.0
Y_train = Y_train.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0
Y_val = Y_val.astype('float32') / 255.0

# Entrenamiento
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    batch_size=64,
    epochs=30,
    callbacks=[checkpoint]
)
