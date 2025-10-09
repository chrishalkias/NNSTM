from dataset import fetch_mnist, transform_label_to_gaussian, g_tester
from network import SimpleFullyConnectedNetwork, untrained_net
import numpy as np
import matplotlib.pyplot as plt
from trainer import regress, plot_training, plot_prediction
from tensorflow import keras

# Initialize and store the dataset
X, y, X_test, y_test = fetch_mnist()
y_transformed = np.array([transform_label_to_gaussian(label, std_dev = 0.001) for label in y])
y_test_transformed = np.array([transform_label_to_gaussian(label, std_dev = 0.001) for label in y_test])

# Gaussian transform the labels
tester = g_tester(y_transformed)

# See the mean output and KDE of an ensemble of neural nets
high_temp = untrained_net(X=X, 
                          ensemble_size=100, 
                          temperature = 10)

low_temp = untrained_net(X=X, 
                         ensemble_size=100, 
                         temperature = 1)

# Now train the model
model, history = regress(X=X, 
                        X_test=X_test, 
                        y_transformed=y_transformed, 
                        y_test_transformed = y_test_transformed, 
                        epochs = 5, 
                        g=0.7)

# Store the relevant quantitites
prediction = np.array(model.predict(X))
loss = history.history['loss']
kld = history.history['kl_divergence']
acc = history.history['accuracy']
epochs = range(1, len(loss) + 1)

# Plot the training metrics
plot_training(epochs=epochs, 
              loss=loss, 
              kld=kld, 
              acc=acc)

# Plot a prediction
plot_prediction(X_test=X_test, 
                y_test=y_test, 
                y_test_transformed=y_test_transformed, 
                prediction=prediction, 
                tests = 1)

# except:
#   model = keras.models.load_model("100_epochs_lin.keras") #load the model if saved
#   prediction = np.array(model.predict(X))
