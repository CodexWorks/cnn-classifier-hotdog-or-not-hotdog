import os

import numpy as np
import tensorflow as tf
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

from src.modelling.MainCNNRecognition import MainCNNRecognition
from src.processing.preparing_image import load_dataset, post_process_data
from src.utilities.constants import dataset_path_images

# -------------------------------------------------------------------------------------------------------------------
# START POINT
# Training + Validation + Ploting

hotdogs = []
not_hotdogs = []

for food_category in os.listdir(dataset_path_images):
    if food_category == "hot_dog":
        # Hotdog - class 0
        for hotdog_img in os.listdir(dataset_path_images + "hot_dog/"):
            hotdogs.append(dataset_path_images + "hot_dog/" + hotdog_img)
    else:
        # not-Hotdog - class 1
        k = 0
        for not_hotdog_img in os.listdir(dataset_path_images + food_category):
            if k >= 200:
                break
            not_hotdogs.append(dataset_path_images + food_category + "/" + not_hotdog_img)
            k = k + 1

print("Hotdog images", len(hotdogs))
print("Not hotdog images", len(not_hotdogs))

# Print two images using cv2_imshow
# Check: https://www.geeksforgeeks.org/python-opencv-cv2-imshow-method/
# print("Hotdog")
# hotdog_image = cv2.imread(hotdogs[5])
# hotdog_image = cv2.resize(hotdog_image, (256, 256))
# cv2.normalize(hotdog_image, hotdog_image, 0, 255, cv2.NORM_MINMAX)
# # cv2_imshow(hotdog_image)
#
# print("Not Hotdog")
# not_hotdog_image = cv2.imread(not_hotdogs[5])
# not_hotdog_image = cv2.resize(not_hotdog_image, (256, 256))
# cv2.normalize(not_hotdog_image, not_hotdog_image, 0, 255, cv2.NORM_MINMAX)
# cv2_imshow(not_hotdog_image)

# All the images will be resized! Set a specific image size
# This value will influence the training process
image_size = 64

X_images, Y_labels = load_dataset(image_size, hotdogs, not_hotdogs)
X_images = post_process_data(X_images)
print(X_images.shape)
print(Y_labels.shape)
Y_labels = to_categorical(Y_labels)

# Define a random state value
# If you set random_state = 42 then no matter how many times you execute
# your code, the result would be the same
# (same values in train and test datasets)
rand_state = 42
tf.random.set_seed(rand_state)
np.random.seed(rand_state)

# Split X_images into X_train and X_test and Y_labes into Y_train and Y_test
X_train, X_test, Y_train, Y_test = train_test_split(X_images, Y_labels,
                                                    test_size=0.10,
                                                    random_state=rand_state)

# print(X_train[1])
print("X_train shape: ", X_train.shape)
print("Y_train shape: ", Y_train.shape)
print("X_test shape: ", X_test.shape)
print("Y_test shape: ", Y_test.shape)

input_shape = (image_size, image_size, 1)

# augmentationImageGenerator = ImageDataGenerator(rotation_range=10,
#                                                 zoom_range=0.15,
#                                                 width_shift_range=0.2,
#                                                 height_shift_range=0.2,
#                                                 shear_range=0.15,
#                                                 horizontal_flip=True,
#                                                 fill_mode="nearest")

cnnRecognition = MainCNNRecognition("cnn_hotdog_v18_50_50_2806", input_shape)
print(cnnRecognition)

cnnRecognition.train(X_train, Y_train, X_test, Y_test)
cnnRecognition.evaluate(X_train, Y_train, X_test, Y_test)
cnnRecognition.save_model()
cnnRecognition.confusion_matrix(X_test, Y_test)
cnnRecognition.plot_history()
