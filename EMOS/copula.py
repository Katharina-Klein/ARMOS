import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfpd = tfp.distributions

T = 49
EPS = 0.1
WEIGHTS = np.array([[2**(1 - abs(i - j)) for i in range(T)] for j in range(T)])

def find_similar_past_obs(tag, forecast, past_tag, past_forecast, past_obs, n_samples=40):
    tag = tag.numpy().astype('str')
    past_tag = past_tag.numpy().astype('str')

    idxs = []
    tags = []
    for i, t in enumerate(tag):
        temp = []
        temp_tag = []
        st = t[-3:]
        ind = [j for j, t2 in enumerate(past_tag) if '_' + st in t2]
        st_obs = tf.gather(past_obs, ind)
        st_fct = tf.gather(past_forecast, ind)
        st_tags = tf.gather(past_tag, ind)
        sim = np.zeros(len(ind))
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
    n_samples = samples.shape[0]

    ref_samples = find_similar_past_obs(tag, forecast, past_tag, past_forecast, past_obs, n_samples)
    
    shuffled_samples = rank_shuffling(samples, ref_samples)

    return shuffled_samples





