from prior import ZVP
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

def manual_NN(weights):
  """
  Sets up a neural net I/O mapping (have to run the NN first!!)

  Inputs:
    weights (np.array) : The trained weights from running backpropagation
  Returns:
    layer2  (np.array) : An array of the last layer preactivations
  """
  nn = ZVP()
  x = (nn.mnist('x')[0]).flatten()
  #print(f"Dims check {len((np.array(weights['w1'])).flatten()) == len(x)**2}")
  layer1 = np.dot(np.array(weights['w1']), x)
  layer2 = np.dot(np.array(weights['w2']), layer1.flatten())
  return layer2
  # layer3 = np.dot(np.array(weights['w3']), layer2.flatten())
  # return layer3

def distribution(weights):
  '''
  Calls manual_NN & Creates prior densities

  Inputs:
    weights (np.array) : The trained weights from running backpropagation
  Returns:
    final  (np.array)  : The final array gaussian KDE distribution
  '''
  from scipy.stats import gaussian_kde
  final = manual_NN(weights).flatten() #call manual_NN
  dist_space = np.linspace(0, max(final), 100)
  kde = gaussian_kde(final)
  plt.title(f'KDE for Probability distribution of $h_2$ for N={28*28}')
  plt.xlabel(r'$|{h_d}|$')
  plt.ylabel('count')
  plt.plot(dist_space, kde(dist_space))
  plt.savefig('figures/annealed_prior.png')
  plt.show()
  #return kde(dist_space)
  return final

def distinguish_PD(p, q):
  """
  Use the KLD to distinguish between the prior computed by the backpropagation
  and the one computed via the analytical solution

  Inputs:
    p (np.array) : The exact prior
    q (np.array) : The annealed prior found through backpropagation

  Returns:
    sp.spatial.distance.jensenshannon(p, q) (float) : The distance measure
  """
  return sp.spatial.distance.jensenshannon(p, q)