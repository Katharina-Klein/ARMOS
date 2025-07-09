import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

class EMOS:
    def __init__(self, num_var, T=49):
        self.initializer_bias = tf.keras.initializers.TruncatedNormal(mean=6.303742072040823, stddev=1)
        self.initializer = tf.keras.initializers.TruncatedNormal(mean=1, stddev=0.5)
        self.a   = tf.Variable(self.initializer_bias(shape=(T,)))
        self.b   = tf.Variable(self.initializer(shape=(T,num_var)))
        self.c   = tf.Variable(self.initializer(shape=(T,)))
        self.d   = tf.Variable(self.initializer(shape=(T,)))
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.05, clipvalue=1.0)
    
    def get_params(self):
        return self.a, self.b, self.c, self.d
    
    def set_params(self, params):
        self.a = params[0]
        self.b = params[1]
        self.c = params[2]
        self.d = params[3]
    
    def get_dist_params(self, x, S2):
        loc = self.a + tf.math.reduce_sum(self.b * x, axis=-1)
        scale = tf.sqrt(self.c + self.d * S2)
        return loc, scale
    
    def loss(self, x, S2, obs):
        loc = self.a + tf.math.reduce_sum(self.b * x, axis=-1)
        scale = tf.sqrt(self.c + self.d * S2)
        dist = tfp.distributions.TruncatedNormal(loc, scale, low=0.0, high=10000000)
        return - dist.log_prob(obs)
    
    def fit(self, x, S2, obs, steps=1000):
        hist = []
        for _ in range(steps):
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(self.loss(x, S2, obs))
            self.opt.minimize(loss, [self.a,self.b,self.c,self.d], tape=tape)
            hist.append(loss.numpy())
        return hist
    
    def sample(self, x, S2, n_samples=30, random=False):
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
        loc = self.a + tf.math.reduce_sum(self.b * x, axis=-1)
        scale = tf.sqrt(self.c + self.d * S2)
        samples = np.zeros((n_samples,) + loc.shape)
        dist = tfp.distributions.TruncatedNormal(loc, scale, low=0.0, high=10000000)
        for i in range(n_samples):
            samples[i] = dist.quantile((i+1) / (n_samples + 1))
        return samples
    


