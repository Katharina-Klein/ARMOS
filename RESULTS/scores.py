import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp



T = 49

WEIGHTS_np = np.array([[2**(1 - abs(i - j)) for i in range(T)] for j in range(T)])
WEIGHTS_tf = tf.constant([[2**(1 - abs(i - j)) for i in range(T)] for j in range(T)])


# Log score for ARMOS
def log_score_armos(model, x, S2, obs):
    loc = model.a + tf.math.reduce_sum(model.b * x, axis=-1)
    scale = tf.sqrt(model.c + model.d * S2)

    res   = obs - loc
    corr  = tf.zeros(res.shape[:-1] + (model.T,))
    for i in range(model.p):
        corr = corr + tf.concat([tf.zeros([res.shape[0],model.p-i]), model.phi[i] * res[...,:model.T-model.p+i]],-1)
    loc   = loc + corr
    
    dist = tfp.distributions.TruncatedNormal(loc, scale, low=0.0, high=10000000)

    return - dist.log_prob(obs)



def energy_score_base(samples, obs, tensor=False):
    if tensor:
        E_1 = tf.linalg.norm(samples - obs, axis=-1)
        E_2 = tf.linalg.norm(tf.expand_dims(samples, 0) - tf.expand_dims(samples, 1), axis=-1)
        ES = tf.reduce_mean(E_1, axis=0) - 0.5 * tf.reduce_mean(E_2, axis=(0,1))
    else:
        E_1 = np.linalg.norm(samples - obs, axis=-1)
        E_2 = np.linalg.norm(np.expand_dims(samples, 0) - np.expand_dims(samples, 1), axis=-1)
        ES = E_1.mean(0) - 0.5 * E_2.mean((0,1))
    return ES

def energy_score(samples, obs, tensor=False, batchsize=None):
    if batchsize == None:
        ES = energy_score_base(samples, obs, tensor)
    else:
        n_days = samples.shape[1]
        n_batches = (n_days // batchsize) + 1
        ES = []
        if tensor:
            for s in range(n_batches):
                start = s*batchsize
                end = min((s+1)*batchsize, n_days)
                ES = tf.concatenate([ES, energy_score_base(samples[:,start:end,:], obs[start:end,:], tensor)], axis = 0)
        else:
            for s in range(n_batches):
                start = s*batchsize
                end = min((s+1)*batchsize, n_days)
                ES = np.concatenate([ES, energy_score_base(samples[:,start:end,:], obs[start:end,:], tensor)], axis=0)
    return ES


def variogram_score_tf(samples, obs, p=0.5, w=WEIGHTS_tf):
    D1 = tf.math.abs(obs[...,None] - obs[...,None,:])**p #(batch, T, T)
    D2 = tf.math.abs(samples[...,None] - samples[...,None,:])**p #(n_samples, batch, T, T)
    VS = (D1 - tf.reduce_mean(D2, axis=0))**2
    VS  = w * tf.linalg.band_part(VS, 0, -1)
    VS = tf.reduce_sum(VS, axis=(-2,-1))
    return VS


def variogram_score_np(samples, obs, p=0.5, w=WEIGHTS_np):
    VS  =  np.abs(obs[..., None] - obs[..., None, :])**p
    VS -= (np.abs(samples[..., None] - samples[..., None, :])**p).mean(axis=0)
    VS  = w * np.triu(VS**2)
    VS  = VS.sum(axis=(-2,-1))    
    return VS


def variogram_score(samples, obs, p=0.5, w=WEIGHTS_np, batchsize=None):
    if batchsize == None:
        VS = variogram_score_np(samples, obs, p, w)
    else:
        n_days = samples.shape[1]
        n_batches = (n_days // batchsize) + 1
        VS = []
        for s in range(n_batches):
            start = s*batchsize
            end = min((s+1)*batchsize, n_days)
            VS = np.concatenate([VS, variogram_score_np(samples[:,start:end,:], obs[start:end,:], p, w)], axis=0)
        VS = VS
    return VS


def bootstrap_choice(test_days, test_tag, test_S2, test_var, test_obs):

    sample_days = np.random.choice(test_days, size=len(test_days), replace=True)

    sample_tags = []

    for day in sample_days:
        for i, t in enumerate(test_tag.numpy().astype('str')):
            if day in t:
                sample_tags.append(i)

    ind = [i for i in sample_tags]

    sample_S2  = tf.gather(test_S2 , ind) 
    sample_var = tf.gather(test_var, ind)
    sample_obs = tf.gather(test_obs, ind)

    return ind, sample_tags, sample_var, sample_S2, sample_obs


def bootstrap(data, test_days, test_tag, test_S2, test_var, test_obs, num_samples=200):

    values = np.zeros(num_samples)

    for k in range(num_samples):
        ind = bootstrap_choice(test_days, test_tag, test_S2, test_var, test_obs)[0]
        values[k] = np.mean(data[ind])

    return np.concatenate([[np.mean(values)], np.percentile(values, [2.5, 97.5])])



