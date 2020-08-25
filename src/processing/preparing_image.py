import cv2
import numpy as np
from skimage import exposure


def rotate_image(image, angle):
    (rows, cols, _) = image.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)

    return cv2.warpAffine(image, M, (cols, rows))


def modify_image(image_path, image_size):
    image = cv2.imread(image_path)
    # Generate a random image
    random_angle = np.random.randint(0, 360)

    # Rotate
    image = rotate_image(image, random_angle)

    # Blur
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # Resize
    image = cv2.resize(image, image_size)

    return image


def load_and_process_image(filepath, image_size):
    """
    Load image using a filepath, and then process it as follows:
         - resize
         - normalize

    @param: filepath: location of an image
    @param: image_size: tuple object; eq: (32, 32)
    """
    print(filepath)
    image = cv2.imread(filepath)
    try:
        image = cv2.resize(image, image_size)
        image = (image / 255.).astype(np.float32)
    except:
        print("An exception occurred")

    return image


def load_images(image_paths, class_label, image_size):
    """
    Go through each path corresponding to the images,
    save the uploaded image and its class label
    """
    x = []
    y = []

    for path in image_paths:
        image = load_and_process_image(path, image_size)
        x.append(image)
        y.append(class_label)

    # Data augmentation
    while len(x) < 25000:
        print("image augmentation - random image path (hotdog)")
        randIdx = np.random.randint(0, len(image_paths))
        image = modify_image(image_paths[randIdx], image_size)
        x.append(image)
        y.append(class_label)

    return x, y


def convert_to_grayscale(images):
    '''
    Convert RGB image to grayscale
    Check: https://www.mathworks.com/help/matlab/ref/rgb2gray.html
    Formula: 0.2989 * R + 0.5870 * G + 0.1140 * B
    '''
    images = 0.2989 * images[:, :, :, 0] + 0.5870 * images[:, :, :, 1] + 0.1140 * images[:, :, :, 2]
    return images


def post_process_data(images):
    '''
    Processes the initial data set, using various auxiliary methods
    '''
    grayImages = convert_to_grayscale(images)

    # Increase constrast
    for i in range(images.shape[0]):
        grayImages[i] = exposure.equalize_hist(grayImages[i])

    # Reshape array to a specific input shape
    grayImages = grayImages.reshape(grayImages.shape + (1,))
    return grayImages


def load_dataset(image_size, hotdogs, not_hotdogs):
    '''
    Load the dataset and separate the images
    (hotdog/not-hotdog) and the corresponding class labels (0/1)
    '''
    image_size = (image_size, image_size)
    x_hotdog, y_hotdog = load_images(hotdogs, 0, image_size)
    x_not_hotdog, y_not_hotdog = load_images(not_hotdogs, 1, image_size)

    print("There are", len(x_hotdog), "hotdog images")
    print("There are", len(x_not_hotdog), "not hotdog images")

    # # Convert X, Y to numpy array
    # X = np.concatenate((x_hotdog_augmented, np.array(x_not_hotdog)), axis=0)
    # Y = np.concatenate((y_hotdog_augmented , np.array(y_not_hotdog)), axis=0)

    # Convert X, Y to numpy array
    X = np.array(x_hotdog + x_not_hotdog)
    Y = np.array(y_hotdog + y_not_hotdog)

    return X, Y
