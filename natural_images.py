import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

from dataprep import get_data

data_set = get_data("./data/natural_images")

num_filters = 8
filter_size = 3
pool_size = 2

# Build the model.
model = tf.keras.Sequential([
    Conv2D(num_filters, filter_size, input_shape=(256, 256, 1)),
    MaxPooling2D(pool_size=pool_size),
    Flatten(),
    Dense(10, activation='softmax'),
])
# Compile the model.
model.compile(
    'adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

# Train the model.
model.fit(
    data_set,
    epochs=3,
)
