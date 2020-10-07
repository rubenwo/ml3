import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

from dataprep import get_data

train_ds, val_ds = get_data()

num_filters = 8
filter_size = 3
pool_size = [2, 2]
strides = 2

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
mc = ModelCheckpoint('./models/best_model.h5', monitor='val_loss', mode='min', save_best_only=True)

# Build the model.
model = tf.keras.Sequential([
    Conv2D(num_filters, filter_size, input_shape=(256, 256, 3)),
    MaxPooling2D(pool_size=pool_size, strides=strides),
    Conv2D(num_filters, filter_size, input_shape=(256 / 2, 256 / 2, 3)),
    MaxPooling2D(pool_size=pool_size, strides=strides),
    Conv2D(num_filters, filter_size, input_shape=(256 / 4, 256 / 4, 3)),
    MaxPooling2D(pool_size=pool_size, strides=strides),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(8, activation='softmax'),
])

# Compile the model.
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

# Train the model.
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    shuffle=True,
    callbacks=[es, mc]
)
