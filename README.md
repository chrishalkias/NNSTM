# Statistical mechanics of Neural networks (NN-STM)

This is a repository containing jupyter notebooks used to compare exact analytical solutions for the partition function of _Linear_ neural networks with the ones computed through ordinary backpropagation.

## About the project

According to [Zavatone-Veth & Pehlevan](https://arxiv.org/abs/2104.11734) the prior density distibution of a Linear Bayesian neural network is given by

$$
p_2(\mathbf{h_2}|x) = \frac{1}{{(4\pi\kappa_2^2})^{n_2/2}}\frac{2}{\Gamma(n_1/2)} \left(\frac{||\mathbf{h_2}||}{2\kappa_2}\right)^{(n_1-n_2)/2}K_{(n_1-n_2)/2}\left(\frac{||\mathbf{h_2}||}{2\kappa_2}\right)
$$

For a network of constant width, this simplifies to:

$$
p_2(\mathbf{h_2}|x) = \frac{1}{{(4\pi\kappa_2^2})^{n/2}}\frac{2}{\Gamma(n/2)} K_0\left(\frac{||\mathbf{h_2}||}{2\kappa_2}\right)
$$

where, $\mathbf{h_2}$ is a vector encoding the preactivation of the last layer, $\kappa_2 = 2\pi\sigma_2^2$ is a measure of the standar deviation of the weight intialization of thelast layer, $x$ is a single training set example $\Gamma$ is the [gamma function](https://en.wikipedia.org/wiki/Gamma_function), $K_0$ is the modified [Bessel](https://en.wikipedia.org/wiki/Bessel_function) function of the second kind of order zero.

The notebooks consist of
simple computations of partition functions and a regression based classifiaction attempt of the MNIST dtaset to be compared with analytical (approximate)
results from statistical mechanical methods.

## Additional information

This project was done as part of my MSc degree in Leiden Univrsity under the supervision of prof. Koenraad Schalm. An online version of the thesis draft can be found in the [student thesis repository](https://studenttheses.universiteitleiden.nl/handle/1887/4255089)
