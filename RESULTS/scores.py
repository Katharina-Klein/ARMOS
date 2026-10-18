import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp



# Constants
T = 49 # Total number of lead times (0-48 hours)
WEIGHTS_np = np.array([[2**(1 - abs(i - j)) for i in range(T)] for j in range(T)]) # Weights for Numpy calculation of the variogram score
WEIGHTS_tf = tf.constant([[2**(1 - abs(i - j)) for i in range(T)] for j in range(T)]) # Weights for TensorFlow calculation of the variogram score



### Scoring rules

def log_score_armos(model, x, S2, obs):
    """Calculate the multivariate log score for the ARMOS model.
    
    Args:
        x: Tensor containing the predictors used to calculate the location parameter.
        S2: Tensor containing the spatial variance used to calculate the scale parameter.
        obs: Tensor containing the observations.
    
    Returns:
        The element-wise log score of the forecast distribution and observations.
    """
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
    """Calculate the multivariate energy score for samples from the forecast distribution.

    Samples and observations can be either TensorFlow tensors or Numpy arrays, depending on the `tensor` argument.
    
    Args:
        samples: Tensor or Numpy array containing the forecast samples.
        obs: Tensor or Numpy array containing the observations.
        tensor: Boolean indicating whether the inputs are TensorFlow tensors (True) or Numpy arrays (False).
    
    Returns:
        A tensor or numpy array containing the energy score of the forecast distribution and observations for each test case.
    """
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
    """Calculate the multivariate energy score for samples from the forecast distribution.
    
    This function can handle large datasets by processing the data in batches, if a `batchsize` is provided.
    Samples and observations can be either TensorFlow tensors or Numpy arrays, depending on the `tensor` argument.
    
    Args:
        samples: Tensor or Numpy array containing the forecast samples.
        obs: Tensor or Numpy array containing the observations.
        tensor: Boolean indicating whether the inputs are TensorFlow tensors (True) or Numpy arrays (False).
        batchsize: Integer specifying the number of days to process in each batch. If None, the entire dataset is processed at once.
    
    Returns:
        A tensor or numpy array containing the energy score of the forecast distribution and observations for each test case.
    """
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
    """Calculate the multivariate variogram score for samples from the forecast distribution.
    
    This function uses TensorFlow tensors for the calculations.
    
    Args:
        samples: Tensor containing the forecast samples.
        obs: Tensor containing the observations.
        p: Float specifying the order of the variogram score (default is 0.5).
        w: Tensor containing the weights for the variogram score calculation (default is WEIGHTS_tf).
    
    Returns:
        A tensor containing the variogram score of the forecast distribution and observations for each test case.
    """
    D1 = tf.math.abs(obs[...,None] - obs[...,None,:])**p #(batch, T, T)
    D2 = tf.math.abs(samples[...,None] - samples[...,None,:])**p #(n_samples, batch, T, T)
    VS = (D1 - tf.reduce_mean(D2, axis=0))**2
    VS  = w * tf.linalg.band_part(VS, 0, -1)
    VS = tf.reduce_sum(VS, axis=(-2,-1))
    return VS


def variogram_score_np(samples, obs, p=0.5, w=WEIGHTS_np):
    """Calculate the multivariate variogram score for samples from the forecast distribution.
    
    This function uses Numpy arrays for the calculations.
    
    Args:
        samples: Numpy array containing the forecast samples.
        obs: Numpy array containing the observations.
        p: Float specifying the order of the variogram score (default is 0.5).
        w: Numpy array containing the weights for the variogram score calculation (default is WEIGHTS_np).
    
    Returns:
        A numpy array containing the variogram score of the forecast distribution and observations for each test case.
    """
    VS  =  np.abs(obs[..., None] - obs[..., None, :])**p
    VS -= (np.abs(samples[..., None] - samples[..., None, :])**p).mean(axis=0)
    VS  = w * np.triu(VS**2)
    VS  = VS.sum(axis=(-2,-1))    
    return VS


def variogram_score(samples, obs, p=0.5, w=WEIGHTS_np, batchsize=None):
    """Calculate the multivariate variogram score for samples from the forecast distribution.
    
    This function can handle large datasets by processing the data in batches, if a `batchsize` is provided.
    This function uses Numpy arrays for the calculations.
    
    Args:
        samples: Tensor or Numpy array containing the forecast samples.
        obs: Tensor or Numpy array containing the observations.
        p: Float specifying the order of the variogram score (default is 0.5).
        w: Numpy array containing the weights for the variogram score calculation (default is WEIGHTS_np).
        batchsize: Integer specifying the number of days to process in each batch. If None, the entire dataset is processed at once.
    
    Returns:
        A tensor or numpy array containing the variogram score of the forecast distribution and observations for each test case.
    """
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




### Bootstrap functions for confidence intervals

def bootstrap_choice(test_days, test_tag, test_S2, test_var, test_obs):
    """Generate bootstrap samples from the test dataset for confidence interval estimation.
    
    Args:
        test_days: Array containing the unique days in the test dataset.
        test_tag: Tensor containing the tags for each test case, indicating the corresponding day.
        test_S2: Tensor containing the spatial variance (used to calculate the scale parameter) for each test case.
        test_var: Tensor containing the predictors (used to calculate the location parameter) for each test case.
        test_obs: Tensor containing the observations for each test case.
    
    Returns:
        A tuple containing the indices of the bootstrap samples, the corresponding tags, and the sampled spatial variance, predictors, and observations.
    """

    # Randomly select days from the test dataset with replacement
    sample_days = np.random.choice(test_days, size=len(test_days), replace=True)

    sample_tags = []

    # For each randomly selected day, find all test cases corresponding to that day and collect their tags
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
    """Estimate the mean and 95% confidence intervals of a score using block bootstrap sampling from the test dataset.
    
    Args:
        data: Numpy array containing the score values for each test case.
        test_days: Array containing the unique days in the test dataset.
        test_tag: Tensor containing the tags for each test case, indicating the corresponding day.
        test_S2: Tensor containing the spatial variance (used to calculate the scale parameter) for each test case.
        test_var: Tensor containing the predictors (used to calculate the location parameter) for each test case.
        test_obs: Tensor containing the observations for each test case.
        num_samples: Integer specifying the number of bootstrap samples to generate (default is 200).
    
    Returns:
        A tuple containing the estimated mean and the 95% confidence interval.
    """

    values = np.zeros(num_samples)

    for k in range(num_samples):
        ind = bootstrap_choice(test_days, test_tag, test_S2, test_var, test_obs)[0]
        values[k] = np.mean(data[ind])

    return np.concatenate([[np.mean(values)], np.percentile(values, [2.5, 97.5])])



