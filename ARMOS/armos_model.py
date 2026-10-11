"""
Implementation of the Autoregressive Model Output Statistics (ARMOS) model for different loss functions.
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class ARMOS_LS:
    """ARMOS model trained on the log score.
    
    The model estimates the parameters of a truncated normal distribution, conditional on previous lead times.
    Its location parameter is a linear function of the predictors, while its scale parameter
    is a linear function of the spatial variance.

    Args:
        num_var: Number of predictors used to compute the location parameter.
        p: Number of previous lead times used to shift the location parameter.
        T: Number of lead times for which to fit the model. Defaults to 49.
    """
    def __init__(self, num_var, p, T=49):
        """Initialize the model parameters and optimizer."""
        self.initializer_bias = tf.keras.initializers.TruncatedNormal(mean=6.303742072040823, stddev=1)  # initialized to the mean wind speed of the training data
        self.initializer = tf.keras.initializers.TruncatedNormal(mean=1, stddev=0.5)
        self.T   = T
        self.p   = p
        self.a   = tf.Variable(self.initializer_bias(shape=(T,)))
        self.b   = tf.Variable(self.initializer(shape=(T,num_var)))
        self.c   = tf.Variable(self.initializer(shape=(T,)))
        self.d   = tf.Variable(self.initializer(shape=(T,)))
        self.phi = tf.Variable(tf.zeros(p))
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.05, clipvalue=1.0)
    
    def get_params(self):
        """Return a tuple containing the trainable model parameters."""
        return self.a, self.b, self.c, self.d, self.phi

    def set_params(self, params):
        """Set the model parameters to known values.
        
        Args:
            params: A tuple containing the replacement parameters a, b, c, d, and phi (in that order).
        """
        self.a = params[0]
        self.b = params[1]
        self.c = params[2]
        self.d = params[3]
        self.phi = params[4]
    
    def get_dist_params(self, x, S2):
        """Calculate the location and scale of the predictive distribution, assuming no shift in the location parameter.
        
        Args:
            x: Tensor containing the predictors used to calculate the location parameter.
            S2: Tensor containing the spatial variance used to calculate the scale parameter.

        Returns:
            A tuple containing the location and scale parameters of the truncated normal distribution.
        """
        loc = self.a + tf.math.reduce_sum(self.b * x, axis=-1)
        scale = tf.sqrt(self.c + self.d * S2)
        phi = self.phi
        return loc, scale, phi
    
    def loss(self, x, S2, obs):
        """Calculate the multivariate log score.
        
        Args:
            x: Tensor containing the predictors used to calculate the location parameter.
            S2: Tensor containing the spatial variance used to calculate the scale parameter.
            obs: Tensor containing the observations.

        Returns:
            The element-wise log score of the forecast distribution and observations.
        """
        # Determine the initial location and scale parameters based on the predictors and spatial variance
        loc = self.a + tf.math.reduce_sum(self.b * x, axis=-1)
        scale = tf.sqrt(self.c + self.d * S2)

        # Calculate the location shift based on the residuals of previous lead times
        res   = obs - loc
        corr  = tf.zeros(res.shape[:-1] + (self.T,))
        for i in range(self.p):
            corr = corr + tf.concat([tf.zeros([res.shape[0],self.p-i]), self.phi[i] * res[...,:self.T-self.p+i]],-1)
        loc   = loc + corr   # Shifted location parameter

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
            self.opt.minimize(loss, [self.a,self.b,self.c,self.d,self.phi], tape=tape)
            hist.append(loss.numpy())
        return hist
    
    def sample(self, x, S2, n_samples=100, random=False):
        """Generate samples from the predictive distribution, lead time by lead time.
        
        Initial samples for lead time t=0 can be drawn randomly or obtained by evaluating equidistant quantiles.

        Args:
            x: Tensor containing the predictors used to calculate the location parameter.
            S2: Tensor containing the spatial variance used to calculate the scale parameter.
            n_samples: Number of samples to generate. Defaults to 100.
            random: If True, draw random samples for lead time t=0. If False, use equidistant quantiles.

        Returns:
            A Numpy array containing the generated samples. The first dimension indexes the samples.
        """

        # Determine the location and scale parameters for all lead times before the location shift
        loc   = self.a + tf.math.reduce_sum(self.b * x, axis=-1)
        scale = tf.sqrt(self.c + self.d * S2)
        samples = np.zeros((n_samples,) + loc.shape)

        # Draw samples from the initial predictive distribution for lead time t=0
        dist = tfp.distributions.TruncatedNormal(loc[...,0], scale[...,0], low=0.0, high=10000000)
        if random:
            samples[...,0] = dist.sample(n_samples).numpy()
        else:
            for i in range(n_samples):
                samples[i,...,0] = dist.quantile((i+1)/(n_samples+1))
        
        # Draw samples for lead times t=1,...,p-1, shifting the location parameter based on the previous t-1 lead times
        for t in range(1, self.p):
            res  = samples[...,0:t] - loc[...,0:t]
            corr = tf.reduce_sum(res * self.phi[self.p-t:self.p], axis=-1)
            dist = tfp.distributions.TruncatedNormal(loc[...,t] + corr, scale[...,t], low=0.0, high=10000000)
            samples[...,t] = dist.sample(1).numpy()

        # Draw samples for lead times t=p,...,T-1, shifting the location parameter based on the previous p lead times
        for t in range(self.p, self.T):
            res  = samples[...,t-self.p:t] - loc[...,t-self.p:t]
            corr = tf.reduce_sum(res * self.phi, axis=-1)
            dist = tfp.distributions.TruncatedNormal(loc[...,t] + corr, scale[...,t], low=0.0, high=10000000)
            samples[...,t] = dist.sample(1).numpy()

        return samples




class ARMOS_MSE:
    """ARMOS model trained on the log score combined with a mean squared error
    between the predicted location parameter and the observations.
        
    The model estimates the parameters of a truncated normal distribution, conditional on previous lead times.
    Its location parameter is a linear function of the predictors, while its scale parameter
    is a linear function of the spatial variance.

    Args:
        num_var: Number of predictors used to compute the location parameter.
        p: Number of previous lead times used to shift the location parameter.
        T: Number of lead times for which to fit the model. Defaults to 49.
    """
    def __init__(self, num_var, p, T=49):
        """Initialize the model parameters and optimizer."""
        self.initializer_bias = tf.keras.initializers.TruncatedNormal(mean=6.303742072040823, stddev=1)  # initialized to the mean wind speed of the training data
        self.initializer = tf.keras.initializers.TruncatedNormal(mean=1, stddev=0.5)
        self.T   = T
        self.p   = p
        self.a   = tf.Variable(self.initializer_bias(shape=(T,)))
        self.b   = tf.Variable(self.initializer(shape=(T,num_var)))
        self.c   = tf.Variable(self.initializer(shape=(T,)))
        self.d   = tf.Variable(self.initializer(shape=(T,)))
        self.phi = tf.Variable(tf.zeros(p))
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.002, clipvalue=1.0)
    
    def get_params(self):
        """Return a tuple containing the trainable model parameters."""
        return self.a, self.b, self.c, self.d, self.phi
    
    def set_params(self, params):
        """Set the model parameters to known values.
        
        Args:
            params: A tuple containing the replacement parameters a, b, c, d, and phi (in that order).
        """
        self.a = params[0]
        self.b = params[1]
        self.c = params[2]
        self.d = params[3]
        self.phi = params[4]
    
    def get_dist_params(self, x, S2):
        """Calculate the location and scale of the predictive distribution, assuming no shift in the location parameter.
        
        Args:
            x: Tensor containing the predictors used to calculate the location parameter.
            S2: Tensor containing the spatial variance used to calculate the scale parameter.
        
        Returns:
            A tuple containing the location and scale parameters of the truncated normal distribution.
        """
        loc = self.a + tf.math.reduce_sum(self.b * x, axis=-1)
        scale = tf.sqrt(self.c + self.d * S2)
        phi = self.phi
        return loc, scale, phi
    
    def loss(self, x, S2, obs):
        """Calculate the multivariate log score, plus the mean squared error between
        the predicted location parameter and the observations.
        
        Args:
            x: Tensor containing the predictors used to calculate the location parameter.
            S2: Tensor containing the spatial variance used to calculate the scale parameter.
            obs: Tensor containing the observations.
        
        Returns:
            The element-wise log score plus mean squared error of the forecast distribution and observations.
        """
        # Determine the initial location and scale parameters based on the predictors and spatial variance
        loc = self.a + tf.math.reduce_sum(self.b * x, axis=-1)
        scale = tf.sqrt(self.c + self.d * S2)

        # Calculate the location shift based on the residuals of previous lead times
        res   = obs - loc
        corr  = tf.zeros(res.shape[:-1] + (self.T,))
        for i in range(self.p):
            corr = corr + tf.concat([tf.zeros([res.shape[0],self.p-i]), self.phi[i] * res[...,:self.T-self.p+i]],-1)
        loc   = loc + corr   # Shifted location parameter

        mu = self.a + tf.math.reduce_sum(self.b * x, axis=-1)   # (Unshifted) predicted location parameter

        dist = tfp.distributions.TruncatedNormal(loc, scale, low=0.0, high=10000000)

        return - dist.log_prob(obs) + (mu - obs)**2
    
    
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
            self.opt.minimize(loss, [self.a,self.b,self.c,self.d,self.phi], tape=tape)
            hist.append(loss.numpy())
        return hist
    
    def sample(self, x, S2, n_samples=100, random=False):
        """Generate samples from the predictive distribution, lead time by lead time.
        
        Initial samples for lead time t=0 can be drawn randomly or obtained by evaluating equidistant quantiles.
        
        Args:
            x: Tensor containing the predictors used to calculate the location parameter.
            S2: Tensor containing the spatial variance used to calculate the scale parameter.
            n_samples: Number of samples to generate. Defaults to 100.
            random: If True, draw random samples for lead time t=0. If False, use equidistant quantiles.
        
        Returns:
            A Numpy array containing the generated samples. The first dimension indexes the samples.
        """
        
        # Determine the location and scale parameters for all lead times before the location shift
        loc   = self.a + tf.math.reduce_sum(self.b * x, axis=-1)
        scale = tf.sqrt(self.c + self.d * S2)
        samples = np.zeros((n_samples,) + loc.shape)

        # Draw samples from the initial predictive distribution for lead time t=0
        dist = tfp.distributions.TruncatedNormal(loc[...,0], scale[...,0], low=0.0, high=10000000)
        if random:
            samples[...,0] = dist.sample(n_samples).numpy()
        else:
            for i in range(n_samples):
                samples[i,...,0] = dist.quantile((i+1)/(n_samples+1))
        
        # Draw samples for lead times t=1,...,p-1, shifting the location parameter based on the previous t-1 lead times
        for t in range(1, self.p):
            res  = samples[...,0:t] - loc[...,0:t]
            corr = tf.reduce_sum(res * self.phi[self.p-t:self.p], axis=-1)
            dist = tfp.distributions.TruncatedNormal(loc[...,t] + corr, scale[...,t], low=0.0, high=10000000)
            samples[...,t] = dist.sample(1).numpy()
        
        # Draw samples for lead times t=p,...,T-1, shifting the location parameter based on the previous p lead times
        for t in range(self.p, self.T):
            res  = samples[...,t-self.p:t] - loc[...,t-self.p:t]
            corr = tf.reduce_sum(res * self.phi, axis=-1)
            dist = tfp.distributions.TruncatedNormal(loc[...,t] + corr, scale[...,t], low=0.0, high=10000000)
            samples[...,t] = dist.sample(1).numpy()

        return samples






class ARMOS_ES:
    """ARMOS model trained on the energy score, or on a combination of the energy score and log score.
        
    The model estimates the parameters of a truncated normal distribution, conditional on previous lead times.
    Its location parameter is a linear function of the predictors, while its scale parameter
    is a linear function of the spatial variance.

    Args:
        num_var: Number of predictors used to compute the location parameter.
        p: Number of previous lead times used to shift the location parameter.
        T: Number of lead times for which to fit the model. Defaults to 49.
        alpha: Weighting factor for the log score in the loss function. Defaults to 0.5.
        n_samples: Number of samples used to approximate the energy score. Defaults to 40.
        batchsize: For batchwise evaluation of the energy score to reduce memory usage. Defaults to 5000.
    """
    def __init__(self, num_var, p, T=49, alpha=0.5, n_samples=40, batchsize=5000):
        """Initialize the model parameters and optimizer."""
        self.initializer_bias = tf.keras.initializers.TruncatedNormal(mean=6.303742072040823, stddev=1)  # initialized to the mean wind speed of the training data
        self.initializer = tf.keras.initializers.TruncatedNormal(mean=1, stddev=0.5)
        self.T   = T
        self.p   = p
        self.alpha = alpha
        self.n_samples = n_samples
        self.batchsize = batchsize
        self.a   = tf.Variable(self.initializer_bias(shape=(T,)))
        self.b   = tf.Variable(self.initializer(shape=(T,num_var)))
        self.c   = tf.Variable(self.initializer(shape=(T,)))
        self.d   = tf.Variable(self.initializer(shape=(T,)))
        self.phi = tf.Variable(tf.zeros(p))
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.01, clipvalue=1.0)
    
    def get_params(self):
        """Return a tuple containing the trainable model parameters."""
        return self.a, self.b, self.c, self.d, self.phi
    
    def set_params(self, params):
        """Set the model parameters to known values.
        
        Args:
            params: A tuple containing the replacement parameters a, b, c, d, and phi (in that order).
        """
        self.a = params[0]
        self.b = params[1]
        self.c = params[2]
        self.d = params[3]
        self.phi = params[4]
    
    def get_dist_params(self, x, S2):
        """Calculate the location and scale of the predictive distribution, assuming no shift in the location parameter.
        
        Args:
            x: Tensor containing the predictors used to calculate the location parameter.
            S2: Tensor containing the spatial variance used to calculate the scale parameter.
        
        Returns:
            A tuple containing the location and scale parameters of the truncated normal distribution.
        """
        loc = self.a + tf.math.reduce_sum(self.b * x, axis=-1)
        scale = tf.sqrt(self.c + self.d * S2)
        phi = self.phi
        return loc, scale, phi
    
    def es(self, x, S2, obs):
        """Calculate the energy score.
        
        Args:
            x: Tensor containing the predictors used to calculate the location parameter.
            S2: Tensor containing the spatial variance used to calculate the scale parameter.
            obs: Tensor containing the observations.
        
        Returns:
            The energy score of the forecast distribution and observations.
        """

        # Determine the initial location and scale parameters based on the predictors and spatial variance
        loc   = self.a + tf.math.reduce_sum(self.b * x, axis=-1)
        scale = tf.sqrt(self.c + self.d * S2)

        ### Draw samples from the predictive distribution in order to approximate the energy score.
        ### Samples are drawn lead time by lead time, shifting the location parameter based on the residuals of previous lead times.
        
        # Draw samples from the initial predictive distribution for lead time t=0
        dist = tfp.distributions.TruncatedNormal(loc[...,0], scale[...,0], low=0.0, high=10000000)

        samples = tf.expand_dims(dist.sample(self.n_samples), axis=-1)

        # Draw samples for lead times t=1,...,p-1, shifting the location parameter based on the previous t-1 lead times
        for t in range(1, self.p):
            res  = samples[...,0:t] - loc[...,0:t]
            corr = tf.reduce_sum(res * self.phi[self.p-t:self.p], axis=-1)
            dist = tfp.distributions.TruncatedNormal(loc[...,t] + corr, scale[...,t], low=0.0, high=10000000)
            samples = tf.concat([samples, tf.transpose(dist.sample(1), [1, 2, 0])], axis=-1)
        
        # Draw samples for lead times t=p,...,T-1, shifting the location parameter based on the previous p lead times
        for t in range(self.p, self.T):
            res  = samples[...,t-self.p:t] - loc[...,t-self.p:t]
            corr = tf.reduce_sum(res * self.phi, axis=-1)
            dist = tfp.distributions.TruncatedNormal(loc[...,t] + corr, scale[...,t], low=0.0, high=10000000)
            samples = tf.concat([samples, tf.transpose(dist.sample(1), [1,2,0])], axis=-1)

        n_days = samples.shape[1]
        n_batch = self.batchsize
        n_batches = (n_days // n_batch) + 1
        ES = 0

        # Calculate the energy score in batches to reduce memory usage
        for s in range(n_batches):
            start = s*n_batch
            end = min((s+1)*n_batch, n_days)
            E_1 = tf.clip_by_value(tf.linalg.norm(samples[:,start:end,:] - obs[start:end,:], axis=-1), 1e-07, 1e10)
            E_2 = tf.clip_by_value(tf.linalg.norm(tf.expand_dims(samples[:,start:end,:], 0) - tf.expand_dims(samples[:,start:end,:], 1), axis=-1), 1e-07, 1e10)
            ES = ES + tf.reduce_sum(tf.reduce_mean(E_1, axis=0) - 0.5 * tf.reduce_mean(E_2, axis=(0,1)), axis=0)

        return ES/n_days
    
    def ls(self, x, S2, obs):
        """Calculate the multivariate log score.
        
        Args:
            x: Tensor containing the predictors used to calculate the location parameter.
            S2: Tensor containing the spatial variance used to calculate the scale parameter.
            obs: Tensor containing the observations.
        
        Returns:
            The element-wise log score of the forecast distribution and observations.
        """
        # Determine the initial location and scale parameters based on the predictors and spatial variance
        loc = self.a + tf.math.reduce_sum(self.b * x, axis=-1)
        scale = tf.sqrt(self.c + self.d * S2)

        # Calculate the location shift based on the residuals of previous lead times
        res   = obs - loc
        corr  = tf.zeros(res.shape[:-1] + (self.T,))
        for i in range(self.p):
            corr = corr + tf.concat([tf.zeros([res.shape[0],self.p-i]), self.phi[i] * res[...,:self.T-self.p+i]],-1)
        loc   = loc + corr   # Shifted location parameter

        dist = tfp.distributions.TruncatedNormal(loc, scale, low=0.0, high=10000000)
        return - dist.log_prob(obs)
    
    def loss(self, x, S2, obs):
        
        return self.es(x, S2, obs) + self.alpha * tf.reduce_mean(self.ls(x, S2, obs), axis=-1)
    
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
            self.opt.minimize(loss, [self.a,self.b,self.c,self.d,self.phi], tape=tape)
            hist.append(loss.numpy())
        return hist
    
    def sample(self, x, S2, n_samples=100, random=False):
        """Generate samples from the predictive distribution, lead time by lead time.
        
        Initial samples for lead time t=0 can be drawn randomly or obtained by evaluating equidistant quantiles.
        
        Args:
            x: Tensor containing the predictors used to calculate the location parameter.
            S2: Tensor containing the spatial variance used to calculate the scale parameter.
            n_samples: Number of samples to generate. Defaults to 100.
            random: If True, draw random samples for lead time t=0. If False, use equidistant quantiles.
        
        Returns:
            A Numpy array containing the generated samples. The first dimension indexes the samples.
        """
        
        # Determine the location and scale parameters for all lead times before the location shift
        loc   = self.a + tf.math.reduce_sum(self.b * x, axis=-1)
        scale = tf.sqrt(self.c + self.d * S2)
        samples = np.zeros((n_samples,) + loc.shape)

        # Draw samples from the initial predictive distribution for lead time t=0
        dist = tfp.distributions.TruncatedNormal(loc[...,0], scale[...,0], low=0.0, high=10000000)
        if random:
            samples[...,0] = dist.sample(n_samples).numpy()
        else:
            for i in range(n_samples):
                samples[i,...,0] = dist.quantile((i+1)/(n_samples+1))
        
        # Draw samples for lead times t=1,...,p-1, shifting the location parameter based on the previous t-1 lead times
        for t in range(1, self.p):
            res  = samples[...,0:t] - loc[...,0:t]
            corr = tf.reduce_sum(res * self.phi[self.p-t:self.p], axis=-1)
            dist = tfp.distributions.TruncatedNormal(loc[...,t] + corr, scale[...,t], low=0.0, high=10000000)
            samples[...,t] = dist.sample(1).numpy()
        
        # Draw samples for lead times t=p,...,T-1, shifting the location parameter based on the previous p lead times
        for t in range(self.p, self.T):
            res  = samples[...,t-self.p:t] - loc[...,t-self.p:t]
            corr = tf.reduce_sum(res * self.phi, axis=-1)
            dist = tfp.distributions.TruncatedNormal(loc[...,t] + corr, scale[...,t], low=0.0, high=10000000)
            samples[...,t] = dist.sample(1).numpy()

        return samples

