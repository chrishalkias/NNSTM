from network import keras_network
import seaborn as sns
import matplotlib.pyplot as plt
from prior import ZVP
import numpy as np
from compare import distribution, distinguish_PD
plt.style.use('ggplot')
sns.set_theme()


# FIX LINE 50 IN THE PRIOR.PY FILE

# Initialize the NN class
experiment = keras_network()

# Train the network and extract the weights
weights = experiment.net(0.01, 2, ["linear","linear","softmax"], plot = True);

# Compute the annealed prior
annealed_prior = distribution(weights)

# Compute the exact prior
exact_prior = ZVP().run()

# First we have to normalize the prior density and the annealed prior
prior_norm = np.abs(exact_prior[1:10]/np.sum(exact_prior[1:10]))
annealed_norm = np.abs(annealed_prior[1:10]/ np.sum(annealed_prior[1:10]))

#Finally compute the KLD of the two distributions
distance = distinguish_PD(prior_norm,annealed_norm)
print(f'Jensen-Shannon divergence between annealed and exact prior = {distance}')
