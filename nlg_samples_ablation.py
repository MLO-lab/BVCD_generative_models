#
import argparse
import os
import pickle
import random

import numpy as np
import torch
import sklearn.metrics as skmetrics
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

from metrics import dist_var, dist_corr, kernel_noise

import config
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--run_id', type=str, default='349oybw1')
parser.add_argument('--embedder_name', type=str, default='e5-large-v2')
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
generation_model = wandb.config.model

def get_kernel_entropies(sequences, embedder_name='all-MiniLM-L6-v2'):

    if embedder_name in ['e5-small-v2', 'e5-large-v2']:
        model_id = 'intfloat/{}'.format(embedder_name)
    elif embedder_name=='gte-large':
        model_id = 'thenlper/gte-large'
    else:
        model_id = 'sentence-transformers/{}'.format(embedder_name)

    embedder = SentenceTransformer(
        model_id, cache_folder=config.data_dir, device=device
    )
    
    n_reps = 10

    results = []
    for sample in tqdm(sequences, desc='Compute sequence embeddings'):
        input = sample['cleaned_generated_texts']

        embeddings = np.array(embedder.encode(input))
        max_gens = embeddings.shape[0]
        for rep_id in range(n_reps):
            for n_gens in range(2, max_gens+1):
                gen_ids = np.random.choice(max_gens, n_gens, replace=False)
                sub_embeddings = embeddings[gen_ids]
                k_entropy_cos = kernel_noise(sub_embeddings, kernel=skmetrics.pairwise.cosine_similarity)
                k_entropy_rbf = kernel_noise(sub_embeddings, kernel=skmetrics.pairwise.rbf_kernel)
                k_entropy_lap = kernel_noise(sub_embeddings, kernel=skmetrics.pairwise.laplacian_kernel)
                k_entropy_pol = kernel_noise(sub_embeddings, kernel=skmetrics.pairwise.polynomial_kernel)
                result = {
                    'k_entropy_cos_{}'.format(embedder_name): k_entropy_cos,
                    'k_entropy_rbf_{}'.format(embedder_name): k_entropy_rbf,
                    'k_entropy_lap_{}'.format(embedder_name): k_entropy_lap,
                    'k_entropy_pol_{}'.format(embedder_name): k_entropy_pol,
                    'id': sample['id'][0],
                    'rep_id': rep_id,
                    'n_gens': n_gens,
                    'rougeL_to_target': sample['rougeL_to_target']
                }
                results.append(result)
    results = pd.DataFrame(results)
    results['id'] = results['id'].astype('object')
    return results

with open(f'{config.output_dir}/{run_name}/{generation_model}_generations.pkl', 'rb') as infile:
    sequences = pickle.load(infile)

results_df = get_kernel_entropies(sequences, embedder_name=args.embedder_name)

auroc_results = []
for rep_id in results_df['rep_id'].unique():
    for n_gens in results_df['n_gens'].unique():
        sub_results = results_df[(results_df['rep_id']==rep_id)&(results_df['n_gens']==n_gens)]
        kent_auroc = skmetrics.roc_auc_score(
            1-(sub_results['rougeL_to_target'] > 0.3).astype('int'),
            sub_results['k_entropy_rbf_{}'.format(args.embedder_name)]
        )
        auroc_results.append({'rep_id': rep_id, 'n_gens': n_gens, 'AUROC': kent_auroc, 'Entropy': 'Kernel'})
auroc_results = pd.DataFrame(auroc_results)

with open(f'results/{args.run_id}_{args.embedder_name}_kernel_entropy_ablation.pkl', 'wb') as outfile:
    pickle.dump(auroc_results, outfile)
