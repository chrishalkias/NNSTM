from tensorflow import keras
import tensorflow as tf
from time import time
import random
import matplotlib.pyplot as plt
import numpy as np

def regress(X, X_test, y_transformed, y_test_transformed, epochs, g):
  sigma_over_N = float(1/784)
  model = keras.models.Sequential()
  init = keras.initializers.RandomNormal(stddev=sigma_over_N)
  flatten = keras.layers.Flatten(input_shape=[28,28])
  model.add(flatten)
  model.add(keras.layers.Dense(units=784, activation='linear', use_bias=False, kernel_regularizer=keras.regularizers.L2(l2=1e-4), name='linear_layer'))
  readout_layer = keras.layers.Dense(units=784, activation =lambda x: x+g*x**3, use_bias=False,kernel_regularizer=keras.regularizers.L2(l2=1e-4), name='readout_layer')
  readout_layer.trainable=True
  model.add(readout_layer)
  #---compile---
  t0 = time()
  model_summary, model_layers  = model.summary(), model.layers
  opt = tf.keras.optimizers.SGD(learning_rate=0.01)
  model.compile(loss='mse', optimizer = opt, metrics=["kl_divergence", "accuracy"])
  history = model.fit(X, y_transformed, epochs=epochs, validation_data =(X_test,y_test_transformed), verbose=1)
  print(f'total runtime {int(time()-t0)/60} minutes')
#   model.save('/model_checkpoints/my_model.keras')
  return model, history

def gather_g_performance_list(X, X_test, y_transformed, y_test_transformed):
  result_list = []
  for g in [0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1]:
    model, history = regress(X, X_test, y_transformed, y_test_transformed, epochs = 2, g=g)
    result_list.append({'g': g, 'accuracy' :history.history['accuracy'], 'val_accuracy' :history.history['val_accuracy']})

def plot_training(epochs, loss, kld, acc):
    plt.plot(epochs, kld, 'b', label='KLD')
    plt.plot(epochs, acc, 'r', label='accuracy')
    plt.plot(epochs, np.ones_like(loss), '--')
    plt.plot(epochs, loss, 'g', label='loss')
    plt.ylim(0,1.5)
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.title(f'Training metrics for {len(epochs)} epochs')
    plt.legend()
    plt.savefig('figures/lin.png')
    plt.show()
  
def plot_prediction(X_test, y_test, y_test_transformed, prediction, tests = 1):
  '''plots {tests} instances of the comparison'''
  for mu in [random.randint(0,len(X_test)) for _ in range(tests)]:
    plt.imshow(X_test[mu])
    plt.title(f'Image of digit {y_test[mu]}')
    plt.show()
    error_squared = round(0.5 * np.power((prediction[mu]-y_test_transformed[mu]).sum(), 2), 3)
    print(f'Prediction Error: {error_squared}')
    digit_space = np.linspace(0,9,784)
    plt.title(f'Trained network prediction for the digit {y_test[mu]}')
    plt.plot(digit_space, prediction[mu], 'b', label = 'Prediction')
    plt.plot(digit_space, y_test_transformed[mu], 'ro', label = 'Target')
    plt.ylim(0,1.1)
    plt.xticks(np.arange(0, 10, step=1))
    plt.legend()
    plt.savefig(f'figures/pred_{mu}[{y_test[mu]}].png')
    plt.show()