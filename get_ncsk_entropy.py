import argparse
import os
import pickle
import random

import numpy as np
import torch
import sklearn.metrics as skmetrics
import pandas as pd
from tqdm import tqdm

from metrics import dist_var, dist_corr, kernel_noise

import config
import wandb

# based on contiguous subsequence kernel (CSK) in https://proceedings.mlr.press/v202/baum23a/baum23a.pdf
def unbroadcasted_unnormalised_contiguous_subsequence_kernel(x, y, t, kernel=None):
    """
    Parameters:
    - x: list or numpy array of shape (lx,)
    - y: list or numpy array of shape (ly,)
    - t: length of subsequences

    Returns:
    - number of matching contiguous subsequences of length t between x and y.
    """
    npx = np.array(x)
    npy = np.array(y)
    lx = npx.shape[-1]
    ly = npy.shape[-1]
    if (t > lx) or (t > ly):
        return 0
    assert t > 0, "t has to be positive integer"
    
    x_indices = np.arange(0, lx - t + 1)
    y_indices = np.arange(0, ly - t + 1)
    
    x_slices = npx[x_indices[:, np.newaxis] + np.arange(t)]
    y_slices = npy[y_indices[:, np.newaxis] + np.arange(t)]

    if kernel is None:
        pairwise_comparison = np.equal(x_slices[:, None], y_slices[None, :])
    else:
        pairwise_comparison = kernel(x_slices[:, None], y_slices[None, :])

    return np.sum(pairwise_comparison)

def unbroadcasted_contiguous_subsequence_kernel(x, y, T=None, kernel=None):
    """
    Parameters:
    - x: list or numpy array of shape (lx,)
    - y: list or numpy array of shape (ly,)
    - t: length of subsequences

    Returns:
    - 'percentage' of matching contiguous subsequences of length t between x and y.
    """
    if T is None:
        T = np.arange(1, np.max([len(x), len(y)])+1).tolist()
    elif not hasattr(T, '__iter__'): #isinstance(t, list):
        #normaliser = 1
        T = [T]

    kxy = np.sum([unbroadcasted_unnormalised_contiguous_subsequence_kernel(x, y, t) for t in T])
    kxx = np.sum([unbroadcasted_unnormalised_contiguous_subsequence_kernel(x, x, t) for t in T])
    kyy = np.sum([unbroadcasted_unnormalised_contiguous_subsequence_kernel(y, y, t) for t in T])
    return kxy / np.sqrt(kxx*kyy)

def contiguous_subsequence_kernel(X, Y, **kwargs):
    """
    Compute the Gram matrix between samples X and Y using the CSK.

    Parameters:
    - X: numpy array of shape (m,) representing the first set of sample sequences.
    - Y: numpy array of shape (n,) representing the second set of sample sequences.
    - T: length or lengths of subsequences; if list then compute avg CSK for all
        lengths in the list; if None compute CSK for all possible lengths with 
        normalisation; !!!Important: never let T depend on the samples!!!

    Returns:
    - Gram matrix of shape (m, n) where the element at position (i, j) is k(X[i], Y[j]).
    """

    # Reshape X and Y for broadcasting
    X_reshaped = X[:, np.newaxis]
    Y_reshaped = Y[np.newaxis, :]

    b = np.broadcast(X_reshaped, Y_reshaped)

    gram_matrix = np.empty(b.shape)
    gram_matrix.flat = [
        unbroadcasted_contiguous_subsequence_kernel(x, y, **kwargs)
        for (x, y) in b
    ]
    return gram_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--generation_model', type=str, default='opt-350m')
parser.add_argument('--run_id', type=str, default='run_1')
parser.add_argument('--T_max', type=int, default=2)
args = parser.parse_args()

device = 'cuda'

# Set a seed value
seed_value = 10
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value

os.environ['PYTHONHASHSEED'] = str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value

random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value

np.random.seed(seed_value)

#Fix torch random seed
torch.manual_seed(seed_value)

os.environ["HF_DATASETS_CACHE"] = config.hf_datasets_cache

wandb.init(project='nlg_uncertainty', id=args.run_id, config=args, resume='allow')
run_name = wandb.run.name

T_max = args.T_max

def get_ncsk_entropies(sequences):

    results = []
    for sample in tqdm(sequences, desc='Compute sequence embeddings'):
        generations = sample['cleaned_generations'].cpu()
        prompt_len = len(sample['prompt'])
        generations = generations[:, prompt_len:]
        gens_var_len = np.empty(generations.shape[:-1], object)
        gens_var_len[...] = [tokens[tokens!=1] for tokens in generations]

        k_entropy_ncsk = kernel_noise(
            gens_var_len,
            kernel=lambda x,y: contiguous_subsequence_kernel(x, y, T=np.arange(1, T_max+1))
        )

        result = {
            'k_entropy_ncsk': k_entropy_ncsk,
            'is_correct': sample['rougeL_to_target'] > 0.3,
            'id': sample['id'][0]
        }
        results.append(result)
    results = pd.DataFrame(results)
    results['id'] = results['id'].astype('object')
    return results

with open(f'{config.output_dir}/{run_name}/{args.generation_model}_generations.pkl', 'rb') as infile:
    sequences = pickle.load(infile)

results_ncsk = get_ncsk_entropies(sequences)

ent_ncsk_auroc = skmetrics.roc_auc_score(
    1-results_ncsk['is_correct'].astype('int'), results_ncsk['k_entropy_ncsk']
)

with open(f'{config.output_dir}/{run_name}/{args.generation_model}_{T_max}csk_auroc.txt', 'w') as outfile:
   outfile.write(f'CSK entropy auroc: {ent_ncsk_auroc}')

#with open(f'{config.output_dir}/{run_name}/{args.generation_model}_kernel_entropies.pkl', 'wb') as outfile:
#    pickle.dump(results_all, outfile)
