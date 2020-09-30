import tensorflow as tf


def get_data(path):
    data_set = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        labels="inferred",
        label_mode="int",
        class_names=None,
        color_mode="rgb",
        batch_size=32,
        image_size=(256, 256),
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation="bilinear",
        follow_links=False,
    )
    return data_set

# print('prepping data...')
#
# image_paths = {}
# # Get list of all image paths
# import os
#
# for root, dirs, files in os.walk("./data/natural_images", topdown=False):
#     print(dirs)
#     print(root)
#     for label in dirs:
#         image_paths[label] = []
#     print(files)
#     # for name in files:
#     #     path = os.path.join(root, name).replace("\\", "/")
#     #     label = path.split("/")[-2]
#     #     print(label)
#     #
#     #     image_paths[label].append(path)
#
# images = {}
#
# for label in image_paths.keys():
#     image_paths[label] = []
#
# # Open images from paths and add to dictionary
#
# for label, paths in image_paths.items():
#     for path in paths:
#         try:
#             img = Image.open(path)
#             images[label].append(img)
#
#         except IOError:
#             print("ERROR")
