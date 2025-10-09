
import  os
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()
from tensorflow import keras

root_logdir = os.path.join(os.curdir,'my_logs')
# Implement normal back propagation for a constant width hidden layer network.
# Create a function that measures the accuracy of the networks and outputs the
# final weights so they can me used later.

class keras_network:
  def __init__(self):
    """
    This class implements a simple neural network to be trained on the MNIST dataset
    
    Methods:
      initializer : Sets the weight initializtion matrix
      net         : Fetches the dataset and creates the neural netowork
    """
    self.scc = "sparse_categorical_crossentropy"
    self.N = 28*28
  def get_run_logdir(self):
    '''Create the logs'''
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir,run_id)

  def initializer(self):
    '''kinda redundand but be sure on the initializations'''
    randy = keras.initializers.RandomNormal(stddev=0.1)
    weight_matrix = [[randy for _ in range(3)]]
    return weight_matrix

  def net(self, lr, epochs, phi, **kwargs):
    '''fetch data, create, compile and run the network'''
    #---dataset---
    mnist = keras.datasets.mnist
    (X_train_full, y_train_full),(X_test,y_test) = mnist.load_data()
    X_valid, X_train = X_train_full[:5000]/255.0, X_train_full[5000:]/255.0
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
    X_test = X_test /255.0
    #---architecture---(3 layers, last layer gets uniform and no update)
    model = keras.models.Sequential()
    flatten = keras.layers.Flatten(input_shape=[28,28])
    layer1 = keras.layers.Dense(units=self.N, activation=phi[0], use_bias=False)
    layer2 = keras.layers.Dense(units=self.N, activation =phi[1], use_bias=False)
    layer3 = keras.layers.Dense(units=10, activation =phi[2], use_bias=False, kernel_initializer='glorot_uniform')
    layer3.trainable=False
    model.add(flatten)
    model.add(layer1)
    model.add(layer2)
    model.add(layer3)
    #---compile---
    model_summary, model_layers  = model.summary(), model.layers
    opt = tf.keras.optimizers.SGD(learning_rate=lr)
    model.compile(loss=self.scc, optimizer = opt, metrics=["accuracy"])
    #---logs---
    run_logdir = self.get_run_logdir()
    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
    history = model.fit(X_train, y_train, epochs=epochs,callbacks = [tensorboard_cb], validation_data =(X_valid, y_valid))
    if kwargs.get("plot") == True:
      pd.DataFrame(history.history).plot(figsize = (8,5))
      plt.grid(True)
      plt.gca().set_ylim(0,1)
      plt.xlabel('Epoch')
      plt.ylabel('Accuracy and Loss')
      plt.title("Training accuracy and Loss of the NN")
      plt.savefig('figures/training.png')
      plt.show()
    #--weights--
    return {'w1': model.layers[1].get_weights(),
            'w2': model.layers[2].get_weights(),
            'w3': model.layers[3].get_weights()}