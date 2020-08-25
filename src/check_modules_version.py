import keras
import matplotlib
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf

print('Numpy version: ' + np.__version__)
print('Pandas version: ' + pd.__version__)
print('Matplotlib version: ' + matplotlib.__version__)
print('Sklearn version: ' + sklearn.__version__)
print('TensorFlow version: ' + tf.__version__)
print('Keras version: ' + keras.__version__)

print(tf.test.is_built_with_cuda())
# Check GPU Physical device
print(tf.config.list_physical_devices('GPU'))
