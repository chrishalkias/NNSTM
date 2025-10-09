import numpy as np
import tensorflow as tf
from scipy.stats import norm
import matplotlib.pyplot as plt

def fetch_mnist():
  '''Loads the MNIST dataset'''
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
  x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0

  return x_train, y_train, x_test, y_test

def transform_label_to_gaussian(y: int, array_size=784, std_dev=1.0) -> np.array:
  '''  Transforms a label y into a 784-dimensional Gaussian array.  '''
  x_values = np.linspace(0, 9, array_size)
  gaussian_array = norm.pdf(x_values, loc=y, scale=std_dev)
  gaussian_array /= gaussian_array.sum()
  return gaussian_array

def g_tester(y_transformed: np.array) -> plt.plot:
  x_axis = np.linspace(0,9,784)
  std_dev=0.1
  for label in range(0,10):
    array = transform_label_to_gaussian(label, std_dev=std_dev)
    plt.plot(x_axis, array, label=f'label={label}')
  plt.title(f"Transformed Label as a Discretized Gaussian, σ={std_dev}")
  plt.xlabel("Target Index (y)")
  plt.ylabel("Gaussian Value")
  plt.legend()
  plt.savefig('figures/gaussian.png')
  plt.show()