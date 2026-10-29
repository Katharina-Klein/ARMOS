"""
Implementation of the similarity-based Schaake Shuffle algorithm
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfpd = tfp.distributions


def find_similar_past_obs(tag, forecast, past_tag, past_forecast, past_obs, n_samples=40):
    """ Determine the most similar past observations based on the similarity of their forecasts to past forecasts.

    Args:
        tag: Tensor containing the tags of the current forecasts.
        forecast: Tensor containing the current forecasts.
        past_tag: Tensor containing the tags of the past forecasts.
        past_forecast: Tensor containing the past forecasts.
        past_obs: Tensor containing the past observations.
        n_samples: Number of similar past observations to find. Defaults to 40.
    
    Returns:
        A tuple containing the most similar past observations and their corresponding tags.
        The first dimension of each array indexes the samples.
    """
    tag = tag.numpy().astype('str')
    past_tag = past_tag.numpy().astype('str')

    idxs = []
    tags = []
    for i, t in enumerate(tag):
        temp = []
        temp_tag = []

        # Find the indices of past forecasts for the same station as the current forecast
        st = t[-3:]
        ind = [j for j, t2 in enumerate(past_tag) if '_' + st in t2]
        st_obs = tf.gather(past_obs, ind)
        st_fct = tf.gather(past_forecast, ind)
        st_tags = tf.gather(past_tag, ind)
        sim = np.zeros(len(ind))

        # Find the most similar past observations based on the similarity of their forecasts to the current forecast
        for k, fct in enumerate(st_fct):
            sim[k] = np.linalg.norm(forecast[i] - fct)
        sim_sort = np.argsort(sim)
        for id in sim_sort[:n_samples]:
            temp.append(st_obs[id])
            temp_tag.append(st_tags[id])
        idxs.append(temp)
        tags.append(temp_tag)
        
    idx = np.array(idxs).transpose([1,0,2])
    tags = np.array(tags).transpose()
    
    return idx, tags


def rank_shuffling(samples, ref_samples):
    """ Shuffle the samples based on the ranks of the reference samples.

    Can be used instead of sim_schaake_shuffle if the reference samples are already known.

    Args:
        samples: Tensor containing the samples to be shuffled.
        ref_samples: Tensor containing the reference samples used to determine the ranks.
    
    Returns:
        A Numpy array containing the shuffled samples. The first dimension indexes the samples.
    """
    sort = np.sort(samples, axis=0)

    y = np.array(ref_samples)
    perm = y.argsort(0)

    shuff = np.zeros(sort.shape)
    idx = np.indices(sort.shape)
    idx[0] = perm
    idx = tuple(np.split(idx, idx.shape[0], axis=0))
    shuff[idx] = sort
    return shuff


def sim_schaake_shuffle(samples, tag, forecast, past_tag, past_forecast, past_obs):
    """ Shuffle the samples based on the ranks of the most similar past observations.
    
    Args:
        samples: Numpy array containing the samples to be shuffled.
        tag: Tensor containing the tags of the current forecasts.
        forecast: Tensor containing the current forecasts.
        past_tag: Tensor containing the tags of the past forecasts.
        past_forecast: Tensor containing the past forecasts.
        past_obs: Tensor containing the past observations.

    Returns:
        A Numpy array containing the shuffled samples. The first dimension indexes the samples.
    """
    n_samples = samples.shape[0]

    ref_samples = find_similar_past_obs(tag, forecast, past_tag, past_forecast, past_obs, n_samples)
    
    shuffled_samples = rank_shuffling(samples, ref_samples)

    return shuffled_samples





