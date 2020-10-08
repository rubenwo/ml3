import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

from dataprep import get_data

train_ds, val_ds = get_data(0.2)

num_filters = 8
filter_size = 3
pool_size = [2, 2]
strides = 2

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
mc = ModelCheckpoint('./models/best_model.h5', monitor='val_loss', mode='min', save_best_only=True,
                     save_weights_only=False)

# Build the model.
model = tf.keras.Sequential([
    # TODO: Big features to smaller features (large filter -> small filter)
    Conv2D(num_filters, filter_size, input_shape=(256, 256, 3)),
    MaxPooling2D(pool_size=pool_size, strides=strides),
    Conv2D(num_filters, filter_size, ),
    MaxPooling2D(pool_size=pool_size, strides=strides),
    Conv2D(num_filters, filter_size, ),
    MaxPooling2D(pool_size=pool_size, strides=strides),
    Conv2D(num_filters, filter_size, ),
    MaxPooling2D(pool_size=pool_size, strides=strides),
    Conv2D(num_filters, filter_size, ),
    MaxPooling2D(pool_size=pool_size, strides=strides),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(8, activation='softmax'),
])

# Compile the model.
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

# Train the model.
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    shuffle=True,
    callbacks=[es, mc]
)
#
# from tensorflow.keras.preprocessing import image
# import numpy as np
#
# img = image.load_img('./data/cat_1.png', grayscale=False, target_size=(256, 256),
#                      color_mode='rgb', interpolation='bilinear')
#
# img_array = image.img_to_array(img)
# print(img_array.dtype)
# print(img_array.shape)
#
# img_array = np.array([img_array])
# predictions = model.predict(img_array)
#
# print(predictions)
