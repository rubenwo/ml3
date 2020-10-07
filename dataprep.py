import tensorflow as tf


def get_data():
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "./data/natural_images",
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(256, 256),
        batch_size=32)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "./data/natural_images",
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(256, 256),
        batch_size=32)

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train_ds, val_ds
