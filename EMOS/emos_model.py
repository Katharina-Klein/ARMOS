"""
Implementation of the Ensemble Model Output Statistics (EMOS) model.
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

class EMOS:
    """Ensemble Model Output Statistics (EMOS) model.

    The model estimates the parameters of a truncated normal distribution.
    Its location parameter is a linear function of the predictors, while its scale parameter
    is a linear function of the spatial variance.

    Args:
        num_var: Number of predictors used to compute the location parameter.
        T: Number of lead times for which to fit the model. Defaults to 49.
    """

    def __init__(self, num_var, T=49):
        """Initialize the model parameters and optimizer."""
        self.initializer_bias = tf.keras.initializers.TruncatedNormal(mean=6.303742072040823, stddev=1)  # initialized to the mean wind speed of the training data
        self.initializer = tf.keras.initializers.TruncatedNormal(mean=1, stddev=0.5)
        self.a   = tf.Variable(self.initializer_bias(shape=(T,)))
        self.b   = tf.Variable(self.initializer(shape=(T,num_var)))
        self.c   = tf.Variable(self.initializer(shape=(T,)))
        self.d   = tf.Variable(self.initializer(shape=(T,)))
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.05, clipvalue=1.0)
    
    def get_params(self):
        """Return a tuple containing the trainable model parameters."""
        return self.a, self.b, self.c, self.d
    
    def set_params(self, params):
        """Set the model parameters to known values.

        Args:
            params: A tuple containing the replacement parameters a, b, c, and d (in that order).
        """
        self.a = params[0]
        self.b = params[1]
        self.c = params[2]
        self.d = params[3]
    
    def get_dist_params(self, x, S2):
        """Calculate the location and scale of the predictive distribution.

        Args:
            x: Tensor containing the predictors used to calculate the location parameter.
            S2: Tensor containing the spatial variance used to calculate the scale parameter.

        Returns:
            A tuple containing the location and scale parameters of the truncated normal distribution.
        """
        loc = self.a + tf.math.reduce_sum(self.b * x, axis=-1)
        scale = tf.sqrt(self.c + self.d * S2)
        return loc, scale
    
    def loss(self, x, S2, obs):
        """Calculate the log score for each of the lead times.

        Args:
            x: Tensor containing the predictors used to calculate the location parameter.
            S2: Tensor containing the spatial variance used to calculate the scale parameter.
            obs: Tensor containing the observations.

        Returns:
            The element-wise log score of the forecast distribution and observations.
        """
        loc = self.a + tf.math.reduce_sum(self.b * x, axis=-1)
        scale = tf.sqrt(self.c + self.d * S2)
        dist = tfp.distributions.TruncatedNormal(loc, scale, low=0.0, high=10000000)
        return - dist.log_prob(obs)
    
    def fit(self, x, S2, obs, steps=1000):
        """Fit the model by minimizing the log score.

        Args:
            x: Tensor containing the predictors used to calculate the location parameter.
            S2: Tensor containing the spatial variance used to calculate the scale parameter.
            obs: Tensor containing the observations.
            steps: Number of optimization steps. Defaults to 1000.

        Returns:
            A list containing the average training loss recorded at each optimization step,
            evaluated before that step's update.
        """
        hist = []
        for _ in range(steps):
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(self.loss(x, S2, obs))
            self.opt.minimize(loss, [self.a,self.b,self.c,self.d], tape=tape)
            hist.append(loss.numpy())
        return hist
    
    def sample(self, x, S2, n_samples=30, random=False):
        """Generate samples from the predictive distribution.

        Samples can be drawn randomly or obtained by evaluating equidistant quantiles.

        Args:
            x: Tensor containing the predictors used to calculate the location parameter.
            S2: Tensor containing the spatial variance used to calculate the scale parameter.
            n_samples: Number of samples to generate. Defaults to 30.
            random: If True, draw random samples. If False, use equidistant quantiles.

        Returns:
            A Numpy array containing the generated samples. The first dimension indexes the samples.
        """
        loc = self.a + tf.math.reduce_sum(self.b * x, axis=-1)
        scale = tf.sqrt(self.c + self.d * S2)
        dist = tfp.distributions.TruncatedNormal(loc, scale, low=0.0, high=10000000)
        if random:
            samples = dist.sample(n_samples).numpy()
        else:
            samples = np.zeros((n_samples,) + loc.shape)
            for i in range(n_samples):
                samples[i] = dist.quantile((i+1) / (n_samples + 1))
        return samples
    
    def sample_quant(self, x, S2, n_samples=30):
        """Generate samples from the predictive distribution using equidistant quantiles.

        Args:
            x: Tensor containing the predictors used to calculate the location parameter.
            S2: Tensor containing the spatial variance used to calculate the scale parameter.
            n_samples: Number of samples to generate. Defaults to 30.

        Returns:
            A Numpy array containing the generated samples. The first dimension indexes the samples.
        """
        loc = self.a + tf.math.reduce_sum(self.b * x, axis=-1)
        scale = tf.sqrt(self.c + self.d * S2)
        samples = np.zeros((n_samples,) + loc.shape)
        dist = tfp.distributions.TruncatedNormal(loc, scale, low=0.0, high=10000000)
        for i in range(n_samples):
            samples[i] = dist.quantile((i+1) / (n_samples + 1))
        return samples
    


