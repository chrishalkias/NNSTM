import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



class SimpleFullyConnectedNetwork:
    def __init__(self, T=1,  input_size=784, linear=True, g=0):
        # Initialize weights and biases for two layers
        self.temp = T
        self.lin = linear
        self.g = g
        self.W1 = np.random.randn(input_size, input_size) * T/input_size
        self.W2 = np.random.randn(input_size, 1) * T/input_size

    def relu(self, x, g=0.01):
        return g*np.sum(x)**3

    def net(self, x):
        # Forward pass through the two layers
        x = x.flatten()  # Flatten the image
        z1 = np.dot(x, self.W1)# First layer linear transformation
        a1 = self.relu(z1)  # Apply ReLU activation
        z2 = np.dot(a1, self.W2)# Second layer linear transformation
        if self.lin == False:
          return self.g*z2**3
        elif self.lin == True:
          return z2
        
def untrained_net(X, ensemble_size=100, temperature=0.1, neuron = 0, mu=0, linear = True):
  global output_list
  output_list = []
  for iterations in range(ensemble_size):
    network = SimpleFullyConnectedNetwork(T=temperature, linear = linear)
    sample_image = X[mu].flatten()
    output = network.net(sample_image)
    output_list.append(np.linalg.norm(output))
    # print(f"Output for the sample image: {output}")
  s = pd.Series(output_list)
  plt.plot(s, 'gx')
  plt.title(f'Outputs of {ensemble_size} untrained NNs for T={temperature}')
  plt.xlabel('Network Index')
  plt.ylabel(f'Output')
  # plt.ylim(0, 1)
  plt.savefig(f'figures/avg_output_for_T={temperature}.png')
  plt.show()
  print()
  s.plot.kde(bw_method=1)
  plt.title(f'Gaussian KDE of outputs_T={temperature}')
  plt.xlim(-1, 1)
  plt.savefig(f'figures/gaussian_kde_T={temperature}.png')
  plt.show()