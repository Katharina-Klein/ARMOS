import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class ARMOS_LS:
    def __init__(self, num_var, p, T=49):
        self.initializer_bias = tf.keras.initializers.TruncatedNormal(mean=6.303742072040823, stddev=1)
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
        return self.a, self.b, self.c, self.d, self.phi

    def set_params(self, params):
        self.a = params[0]
        self.b = params[1]
        self.c = params[2]
        self.d = params[3]
        self.phi = params[4]
    
    def get_dist_params(self, x, S2):
        loc = self.a + tf.math.reduce_sum(self.b * x, axis=-1)
        scale = tf.sqrt(self.c + self.d * S2)
        phi = self.phi
        return loc, scale, phi
    
    def loss(self, x, S2, obs):
        loc = self.a + tf.math.reduce_sum(self.b * x, axis=-1)
        scale = tf.sqrt(self.c + self.d * S2)

        res   = obs - loc
        corr  = tf.zeros(res.shape[:-1] + (self.T,))
        for i in range(self.p):
            corr = corr + tf.concat([tf.zeros([res.shape[0],self.p-i]), self.phi[i] * res[...,:self.T-self.p+i]],-1)
        loc   = loc + corr

        dist = tfp.distributions.TruncatedNormal(loc, scale, low=0.0, high=10000000)
        return - dist.log_prob(obs)
    
    def fit(self, x, S2, obs, steps=1000):
        hist = []
        for _ in range(steps):
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(self.loss(x, S2, obs))
            self.opt.minimize(loss, [self.a,self.b,self.c,self.d,self.phi], tape=tape)
            hist.append(loss.numpy())
        return hist
    
    def sample(self, x, S2, n_samples=100, random=False):
        loc   = self.a + tf.math.reduce_sum(self.b * x, axis=-1)
        scale = tf.sqrt(self.c + self.d * S2)
        samples = np.zeros((n_samples,) + loc.shape)
        dist = tfp.distributions.TruncatedNormal(loc[...,0], scale[...,0], low=0.0, high=10000000)
        if random:
            samples[...,0] = dist.sample(n_samples).numpy()
        else:
            for i in range(n_samples):
                samples[i,...,0] = dist.quantile((i+1)/(n_samples+1))
        for t in range(1, self.p):
            res  = samples[...,0:t] - loc[...,0:t]
            corr = tf.reduce_sum(res * self.phi[self.p-t:self.p], axis=-1)
            dist = tfp.distributions.TruncatedNormal(loc[...,t] + corr, scale[...,t], low=0.0, high=10000000)
            samples[...,t] = dist.sample(1).numpy()
        for t in range(self.p, self.T):
            res  = samples[...,t-self.p:t] - loc[...,t-self.p:t]
            corr = tf.reduce_sum(res * self.phi, axis=-1)
            dist = tfp.distributions.TruncatedNormal(loc[...,t] + corr, scale[...,t], low=0.0, high=10000000)
            samples[...,t] = dist.sample(1).numpy()

        return samples




class ARMOS_MSE:
    def __init__(self, num_var, p, T=49):
        self.initializer_bias = tf.keras.initializers.TruncatedNormal(mean=6.303742072040823, stddev=1)
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
        return self.a, self.b, self.c, self.d, self.phi
    
    def set_params(self, params):
        self.a = params[0]
        self.b = params[1]
        self.c = params[2]
        self.d = params[3]
        self.phi = params[4]
    
    def get_dist_params(self, x, S2):
        loc = self.a + tf.math.reduce_sum(self.b * x, axis=-1)
        scale = tf.sqrt(self.c + self.d * S2)
        phi = self.phi
        return loc, scale, phi
    
    def loss(self, x, S2, obs):
        loc = self.a + tf.math.reduce_sum(self.b * x, axis=-1)
        scale = tf.sqrt(self.c + self.d * S2)

        res   = obs - loc
        corr  = tf.zeros(res.shape[:-1] + (self.T,))
        for i in range(self.p):
            corr = corr + tf.concat([tf.zeros([res.shape[0],self.p-i]), self.phi[i] * res[...,:self.T-self.p+i]],-1)
        loc   = loc + corr

        mu = self.a + tf.math.reduce_sum(self.b * x, axis=-1)

        dist = tfp.distributions.TruncatedNormal(loc, scale, low=0.0, high=10000000)

        return - dist.log_prob(obs) + (mu - obs)**2
    
    
    def fit(self, x, S2, obs, steps=1000):
        hist = []
        for _ in range(steps):
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(self.loss(x, S2, obs))
            self.opt.minimize(loss, [self.a,self.b,self.c,self.d,self.phi], tape=tape)
            hist.append(loss.numpy())
        return hist
    
    def sample(self, x, S2, n_samples=100, random=False):
        loc   = self.a + tf.math.reduce_sum(self.b * x, axis=-1)
        scale = tf.sqrt(self.c + self.d * S2)
        samples = np.zeros((n_samples,) + loc.shape)
        dist = tfp.distributions.TruncatedNormal(loc[...,0], scale[...,0], low=0.0, high=10000000)
        if random:
            samples[...,0] = dist.sample(n_samples).numpy()
        else:
            for i in range(n_samples):
                samples[i,...,0] = dist.quantile((i+1)/(n_samples+1))
        for t in range(1, self.p):
            res  = samples[...,0:t] - loc[...,0:t]
            corr = tf.reduce_sum(res * self.phi[self.p-t:self.p], axis=-1)
            dist = tfp.distributions.TruncatedNormal(loc[...,t] + corr, scale[...,t], low=0.0, high=10000000)
            samples[...,t] = dist.sample(1).numpy()
        for t in range(self.p, self.T):
            res  = samples[...,t-self.p:t] - loc[...,t-self.p:t]
            corr = tf.reduce_sum(res * self.phi, axis=-1)
            dist = tfp.distributions.TruncatedNormal(loc[...,t] + corr, scale[...,t], low=0.0, high=10000000)
            samples[...,t] = dist.sample(1).numpy()

        return samples






class ARMOS_ES:
    def __init__(self, num_var, p, T=49, alpha=0.5, n_samples=40, batchsize=5000):
        self.initializer_bias = tf.keras.initializers.TruncatedNormal(mean=6.303742072040823, stddev=1)
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
        return self.a, self.b, self.c, self.d, self.phi
    
    def set_params(self, params):
        self.a = params[0]
        self.b = params[1]
        self.c = params[2]
        self.d = params[3]
        self.phi = params[4]
    
    def get_dist_params(self, x, S2):
        loc = self.a + tf.math.reduce_sum(self.b * x, axis=-1)
        scale = tf.sqrt(self.c + self.d * S2)
        phi = self.phi
        return loc, scale, phi
    
    def es(self, x, S2, obs):

        loc   = self.a + tf.math.reduce_sum(self.b * x, axis=-1)
        scale = tf.sqrt(self.c + self.d * S2)

        dist = tfp.distributions.TruncatedNormal(loc[...,0], scale[...,0], low=0.0, high=10000000)

        samples = tf.expand_dims(dist.sample(self.n_samples), axis=-1)

        for t in range(1, self.p):
            res  = samples[...,0:t] - loc[...,0:t]
            corr = tf.reduce_sum(res * self.phi[self.p-t:self.p], axis=-1)
            dist = tfp.distributions.TruncatedNormal(loc[...,t] + corr, scale[...,t], low=0.0, high=10000000)
            samples = tf.concat([samples, tf.transpose(dist.sample(1), [1, 2, 0])], axis=-1)
        for t in range(self.p, self.T):
            res  = samples[...,t-self.p:t] - loc[...,t-self.p:t]
            corr = tf.reduce_sum(res * self.phi, axis=-1)
            dist = tfp.distributions.TruncatedNormal(loc[...,t] + corr, scale[...,t], low=0.0, high=10000000)
            samples = tf.concat([samples, tf.transpose(dist.sample(1), [1,2,0])], axis=-1)

        n_days = samples.shape[1]
        n_batch = self.batchsize
        n_batches = (n_days // n_batch) + 1
        ES = 0

        for s in range(n_batches):
            start = s*n_batch
            end = min((s+1)*n_batch, n_days)
            E_1 = tf.clip_by_value(tf.linalg.norm(samples[:,start:end,:] - obs[start:end,:], axis=-1), 1e-07, 1e10)
            E_2 = tf.clip_by_value(tf.linalg.norm(tf.expand_dims(samples[:,start:end,:], 0) - tf.expand_dims(samples[:,start:end,:], 1), axis=-1), 1e-07, 1e10)
            ES = ES + tf.reduce_sum(tf.reduce_mean(E_1, axis=0) - 0.5 * tf.reduce_mean(E_2, axis=(0,1)), axis=0)


        return ES/n_days
    
    def ls(self, x, S2, obs):
        loc = self.a + tf.math.reduce_sum(self.b * x, axis=-1)
        scale = tf.sqrt(self.c + self.d * S2)

        res   = obs - loc
        corr  = tf.zeros(res.shape[:-1] + (self.T,))
        for i in range(self.p):
            corr = corr + tf.concat([tf.zeros([res.shape[0],self.p-i]), self.phi[i] * res[...,:self.T-self.p+i]],-1)
        loc   = loc + corr

        dist = tfp.distributions.TruncatedNormal(loc, scale, low=0.0, high=10000000)
        return - dist.log_prob(obs)
    
    def loss(self, x, S2, obs):
        
        return self.es(x, S2, obs) + self.alpha * tf.reduce_mean(self.ls(x, S2, obs), axis=-1)
    
    def fit(self, x, S2, obs, steps=1000):
        hist = []
        for _ in range(steps):
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(self.loss(x, S2, obs))
            self.opt.minimize(loss, [self.a,self.b,self.c,self.d,self.phi], tape=tape)
            hist.append(loss.numpy())
        return hist
    
    def sample(self, x, S2, n_samples=100, random=False):
        loc   = self.a + tf.math.reduce_sum(self.b * x, axis=-1)
        scale = tf.sqrt(self.c + self.d * S2)
        samples = np.zeros((n_samples,) + loc.shape)
        dist = tfp.distributions.TruncatedNormal(loc[...,0], scale[...,0], low=0.0, high=10000000)
        if random:
            samples[...,0] = dist.sample(n_samples).numpy()
        else:
            for i in range(n_samples):
                samples[i,...,0] = dist.quantile((i+1)/(n_samples+1))
        for t in range(1, self.p):
            res  = samples[...,0:t] - loc[...,0:t]
            corr = tf.reduce_sum(res * self.phi[self.p-t:self.p], axis=-1)
            dist = tfp.distributions.TruncatedNormal(loc[...,t] + corr, scale[...,t], low=0.0, high=10000000)
            samples[...,t] = dist.sample(1).numpy()
        for t in range(self.p, self.T):
            res  = samples[...,t-self.p:t] - loc[...,t-self.p:t]
            corr = tf.reduce_sum(res * self.phi, axis=-1)
            dist = tfp.distributions.TruncatedNormal(loc[...,t] + corr, scale[...,t], low=0.0, high=10000000)
            samples[...,t] = dist.sample(1).numpy()

        return samples

