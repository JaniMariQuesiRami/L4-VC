import numpy as np
from tensorflow.keras import layers, models, optimizers, callbacks

# Cargar el dataset generado en el paso anterior
datos = np.load('ventanas_dataset.npz')
X_train, Y_train = datos['X_train'], datos['Y_train']
X_val, Y_val = datos['X_val'], datos['Y_val']

# Obtener dimensiones de ventana
k = X_train.shape[1]

# ---------------------------------------
# Definir la arquitectura U-Net (pequeña)
# ---------------------------------------

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

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

# ---------------------------------------
# Compilar y entrenar el modelo
# ---------------------------------------

model = unet_model()
model.compile(optimizer=optimizers.Adam(1e-4), loss='mse', metrics=['mae'])

model.summary()

# Callbacks (guardar el mejor modelo)
checkpoint = callbacks.ModelCheckpoint('mejor_unet.h5', monitor='val_loss',
                                       save_best_only=True, verbose=1)

# Entrenamiento
history = model.fit(
    X_train / 255.0, Y_train / 255.0,
    validation_data=(X_val / 255.0, Y_val / 255.0),
    batch_size=64,
    epochs=30,
    callbacks=[checkpoint]
)
