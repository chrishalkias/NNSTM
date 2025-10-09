from tensorflow import keras
import numpy as np
import scipy
import matplotlib.pyplot as plt


class ZVP():
  def __init__(self):
    """
    Implements the exact solution of the two layer neural network
    
    Attributes:
      N       : The network width
      temp    : The temperature of the weight initialization
      sigma   : The standard deviation of the weight initialization

    Methods:
      mnist      : Fetches the dataset and returns the example or the label
      prior      : Computes the analytical prior
      likelihood : Computes the likelihood distibution density
      p_xy       : An (optional) normalization scalar
    """
    self.N=100
    self.temp = 1
    self.sigma = 1/self.N**2 # *self.temp #?

  def mnist(self, boundary:str) -> dict:
    '''loads mnist and outputs a subset of it'''
    mnist = keras.datasets.mnist
    (X_train_full, y_train_full),(X_test,y_test) = mnist.load_data()
    X_train_full = X_train_full / 255.0
    X_test = X_test / 255.0
    if boundary == 'x':
      return X_test
    elif boundary == 'y':
      return y_test

  def prior(self,h:np.array, x:np.array) -> np.array:
    '''computes the prior from the ZVP formula, this is the gist of it'''
    norm_x = np.linalg.norm(x)
    gamma = scipy.special.gamma(self.N/2)
    prefactor = 1 #(4*np.pi*(self.sigma**4)*norm_x**2)**(-self.N/2) * (2/gamma)
    k0 = scipy.special.kn(0, h /(self.sigma * norm_x), out=None)
    #print(f"|x|= {norm_x:.3f}\nΓ(N/2) = {gamma}\nprefactor = {prefactor}\nk0 = {k0[0::10]}")
    return prefactor * k0

  def likelihood(self, y:np.array, h:np.array) -> float:
    '''Computes the likelihood p(y|h,x) = a gaussian centered around y???'''
    #temperature should go in here, but how?
    last_layer_weights = np.random.uniform(self.N) #These we should get from backprop!!!
    softmax = scipy.special.softmax(h)
    f = np.sum(last_layer_weights * softmax)
    loss = 0.5*(f-y)**2
    print(f"True label = {y} \nPredictor value = {f:.3f}\nloss = {loss:.3f}\n")
    return np.exp(-loss)

  def p_xy(self):
    '''normalization'''
    pass
  
  def run(self):
    """
    Brings the class components together to compute the exact prior
    
    Returns:
        p_prior (nd.array) : An array of the prior p(h_d)
    """
    self=ZVP()
    h_d = np.linspace(0,1,784)
    p_prior = self.prior(h_d, self.mnist('x'))

    #---Prints and Plots---
    print(f"Number of neurons per layer: {self.N}")
    print(f"|h_d|/σ²|x|={np.linalg.norm(h_d / (1/self.N**2)*np.linalg.norm(self.mnist('x')))}")
    #print(f"likelihood = {p_likelihood}\n prior = {p_prior[0::25]}\n")
    plt.title(f'Prior density $p_2({{h_d}}| {{x}})$, N={self.N}')
    plt.xlabel(r'$|{h_d}|$')
    plt.ylabel('p_2')
    plt.plot(h_d, p_prior, color="red")
    plt.savefig('figures/prior.png')
    plt.show()
    return p_prior