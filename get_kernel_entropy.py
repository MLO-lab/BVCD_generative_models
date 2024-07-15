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
parser.add_argument('--generation_model', type=str, default='opt-350m')
#parser.add_argument('--evaluation_model', type=str, default='opt-350m')
parser.add_argument('--run_id', type=str, default='run_1')
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

def get_kernel_entropies(sequences, embedder_name='all-MiniLM-L6-v2'):

    if embedder_name=='e5-small-v2':
        model_id = 'intfloat/e5-small-v2'
    elif embedder_name=='gte-large':
        model_id = 'thenlper/gte-large'
    else:
        model_id = 'sentence-transformers/{}'.format(embedder_name)

    embedder = SentenceTransformer(
        model_id, cache_folder=config.data_dir, device=device
    )

    results = []
    for sample in tqdm(sequences, desc='Compute sequence embeddings'):
        input = sample['cleaned_generated_texts']

        embeddings = np.array(embedder.encode(input))
        k_entropy_cos = kernel_noise(embeddings, kernel=skmetrics.pairwise.cosine_similarity)
        k_entropy_rbf = kernel_noise(embeddings, kernel=skmetrics.pairwise.rbf_kernel)
        k_entropy_lap = kernel_noise(embeddings, kernel=skmetrics.pairwise.laplacian_kernel)
        k_entropy_pol = kernel_noise(embeddings, kernel=skmetrics.pairwise.polynomial_kernel)
        result = {
            'embeddings_{}'.format(embedder_name): embeddings,
            'k_entropy_cos_{}'.format(embedder_name): k_entropy_cos,
            'k_entropy_rbf_{}'.format(embedder_name): k_entropy_rbf,
            'k_entropy_lap_{}'.format(embedder_name): k_entropy_lap,
            'k_entropy_pol_{}'.format(embedder_name): k_entropy_pol,
            'id': sample['id'][0]
        }
        results.append(result)
    results = pd.DataFrame(results)
    results['id'] = results['id'].astype('object')
    return results

with open(f'{config.output_dir}/{run_name}/{args.generation_model}_generations.pkl', 'rb') as infile:
    sequences = pickle.load(infile)

results_all = get_kernel_entropies(sequences, embedder_name='all-MiniLM-L6-v2')
for embedder_name in ['all-mpnet-base-v2', 'all-MiniLM-L12-v2', 'e5-small-v2', 'gte-large']:
    results = get_kernel_entropies(sequences, embedder_name=embedder_name)
    results_all = pd.concat([results_all, results.drop('id', axis=1)], axis=1)

#ent_cos_auroc = skmetrics.roc_auc_score(1-(pd.DataFrame(sequences)['rougeL_to_target'] > 0.3).astype('int'),results['k_entropy_cos'])
#print('Ent Cos Auroc: ', ent_cos_auroc)

with open(f'{config.output_dir}/{run_name}/{args.generation_model}_kernel_entropies.pkl', 'wb') as outfile:
    pickle.dump(results_all, outfile)
