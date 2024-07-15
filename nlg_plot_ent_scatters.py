# parse arguments
import argparse
import json
import pickle

import config
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import sklearn.metrics
import torch
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--run_ids', nargs='+', default=[])
parser.add_argument('--verbose', type=bool, default=False)
args = parser.parse_args()

overall_result_dict = {}

aurocs_across_models = []

sequence_embeddings_dict = {}

run_ids_to_analyze = args.run_ids
for run_id in run_ids_to_analyze:

    wandb.init(project='nlg_uncertainty', id=run_id, resume='allow')
    run_name = wandb.run.name
    model_name = wandb.config.model
    dataset = wandb.config.dataset
    print(run_name)

    def get_similarities_df():
        """Get the similarities df from the pickle file"""
        with open(f'{config.output_dir}/{run_name}/{model_name}_generations_similarities.pkl', 'rb') as f:
            similarities = pickle.load(f)
            similarities_df = pd.DataFrame.from_dict(similarities, orient='index')
            similarities_df['id'] = similarities_df.index
            similarities_df['has_semantically_different_answers'] = similarities_df[
                'has_semantically_different_answers'].astype('int')
            similarities_df['rougeL_among_generations'] = similarities_df['syntactic_similarities'].apply(
                lambda x: x['rougeL'])

            return similarities_df

    def get_variances_df():
        with open(f'{config.output_dir}/{run_name}/{model_name}_kernel_entropies.pkl', 'rb') as infile:
            variances_df = pickle.load(infile)
            #variances_df = pd.DataFrame(variances_df)
            #variances_df['id'] = variances_df['id'].apply(lambda x: x[0])
            #variances_df['id'] = variances_df['id'].astype('object')

            return variances_df

    def get_generations_df():
        """Get the generations df from the pickle file"""
        with open(f'{config.output_dir}/{run_name}/{model_name}_generations.pkl', 'rb') as infile:
            generations = pickle.load(infile)
            generations_df = pd.DataFrame(generations)
            generations_df['id'] = generations_df['id'].apply(lambda x: x[0])
            generations_df['id'] = generations_df['id'].astype('object')
            if not generations_df['semantic_variability_reference_answers'].isnull().values.any():
                generations_df['semantic_variability_reference_answers'] = generations_df[
                    'semantic_variability_reference_answers'].apply(lambda x: x[0].item())

            if not generations_df['rougeL_reference_answers'].isnull().values.any():
                generations_df['rougeL_reference_answers'] = generations_df['rougeL_reference_answers'].apply(
                    lambda x: x[0].item())
            generations_df['length_of_most_likely_generation'] = generations_df['most_likely_generation'].apply(
                lambda x: len(str(x).split(' ')))
            generations_df['length_of_answer'] = generations_df['answer'].apply(lambda x: len(str(x).split(' ')))
            generations_df['variance_of_length_of_generations'] = generations_df['generated_texts'].apply(
                lambda x: np.var([len(str(y).split(' ')) for y in x]))
            generations_df['correct'] = (generations_df['rougeL_to_target'] > 0.3).astype('int')

            return generations_df

    def get_likelihoods_df():
        """Get the likelihoods df from the pickle file"""

        with open(f'{config.output_dir}/{run_name}/aggregated_likelihoods_{model_name}_generations.pkl', 'rb') as f:
            likelihoods = pickle.load(f)
            #print(likelihoods.keys())

            subset_keys = ['average_predictive_entropy_on_subset_' + str(i) for i in range(1, num_generations + 1)]
            subset_keys += ['predictive_entropy_on_subset_' + str(i) for i in range(1, num_generations + 1)]
            subset_keys += ['semantic_predictive_entropy_on_subset_' + str(i) for i in range(1, num_generations + 1)]
            subset_keys += ['number_of_semantic_sets_on_subset_' + str(i) for i in range(1, num_generations + 1)]

            keys_to_use = ('ids', 'predictive_entropy', 'mutual_information', 'average_predictive_entropy',\
                            'average_pointwise_mutual_information', 'average_neg_log_likelihood_of_most_likely_gen',\
                            'average_neg_log_likelihood_of_second_most_likely_gen', 'neg_log_likelihood_of_most_likely_gen',\
                            'predictive_entropy_over_concepts', 'number_of_semantic_sets', 'unnormalised_entropy_over_concepts')

            likelihoods_small = dict((k, likelihoods[k]) for k in keys_to_use + tuple(subset_keys))
            for key in likelihoods_small:
                if key == 'average_predictive_entropy_on_subsets':
                    likelihoods_small[key].shape
                if type(likelihoods_small[key]) is torch.Tensor:
                    likelihoods_small[key] = torch.squeeze(likelihoods_small[key].cpu())

            sequence_embeddings = likelihoods['sequence_embeddings']

            likelihoods_df = pd.DataFrame.from_dict(likelihoods_small)

            likelihoods_df.rename(columns={'ids': 'id'}, inplace=True)

            return likelihoods_df, sequence_embeddings

    similarities_df = get_similarities_df()
    generations_df = get_generations_df()
    variances_df = get_variances_df()
    num_generations = len(generations_df['generated_texts'][0])
    likelihoods_df, sequence_embeddings = get_likelihoods_df()
    result_df = generations_df.merge(similarities_df, on='id')
    result_df = result_df.merge(likelihoods_df, on='id')
    result_df = result_df.merge(variances_df, on='id')

    n_samples_before_filtering = len(result_df)
    result_df['len_most_likely_generation_length'] = result_df['most_likely_generation'].apply(lambda x: len(x.split()))

    # Begin analysis
    result_dict = {}
    result_dict['accuracy'] = result_df['correct'].mean()

    ### Remove NaNs (why wasnt this in original implementation?)
    result_df = result_df[result_df['average_predictive_entropy'].notnull()]

    # scatterplot of kernel entropies (we only plot a limited number of instances)
    max_n_eval = 100
    #samples = np.random.choice(result_df.shape[0], size=max_n_eval, replace=False)
    for key in result_df.keys():
        if 'k_entropy_rbf_e5-small-v2' in key:
            sns.set(font_scale=1.6)
            sns.set_style('ticks')
            plot = sns.scatterplot(y=result_df['rougeL_to_target'][:max_n_eval], x=result_df[key][:max_n_eval])
            plot.axes.tick_params(axis='x', rotation=45)
            sns.despine(plot.figure)
            plot.set_ylabel("RougeL(answer, target)")
            plot.set_xlabel("Kernel entropy RBF e5-small-v2")
            plot.figure.savefig('plots/nlg_scatter_rougeL_{}_{}_{}.svg'.format(dataset, key, model_name), bbox_inches='tight')
            plot.figure.clf()

            sns.set(font_scale=1.6)
            sns.set_style('ticks')
            plot = sns.histplot(y=result_df['rougeL_to_target'], x=result_df[key])
            plot.axes.tick_params(axis='x', rotation=45)
            sns.despine(plot.figure)
            plot.set_ylabel("RougeL(answer, target)")
            plot.set_xlabel("Kernel entropy RBF e5-small-v2")
            plot.figure.savefig('plots/nlg_hist_rougeL_{}_{}_{}.svg'.format(dataset, key, model_name), bbox_inches='tight')
            plot.figure.clf()

            sns.set(font_scale=1.6)
            sns.set_style('ticks')
            plot = sns.kdeplot(y=result_df['rougeL_to_target'], x=result_df[key])
            plot.axes.tick_params(axis='x', rotation=45)
            sns.despine(plot.figure)
            plot.set_ylabel("RougeL(answer, target)")
            plot.set_xlabel("Kernel entropy RBF e5-small-v2")
            plot.figure.savefig('plots/nlg_kde_rougeL_{}_{}_{}.svg'.format(dataset, key, model_name), bbox_inches='tight')
            plot.figure.clf()

            sns.set(font_scale=1.6)
            sns.set_style('ticks')
            plot = sns.scatterplot(y=result_df['rougeL_to_target'][:max_n_eval], x=np.log1p(result_df[key][:max_n_eval]+0.0001))
            plot.axes.tick_params(axis='x', rotation=45)
            sns.despine(plot.figure)
            plot.set_ylabel("RougeL(answer, target)")
            plot.figure.savefig('plots/nlg_scatter_log_rougeL_{}_{}_{}.svg'.format(dataset, key, model_name), bbox_inches='tight')
            plot.figure.clf()

            sns.set(font_scale=1.6)
            sns.set_style('ticks')
            plot = sns.histplot(y=result_df['rougeL_to_target'], x=np.log1p(result_df[key]+0.0001))
            plot.axes.tick_params(axis='x', rotation=45)
            sns.despine(plot.figure)
            plot.set_ylabel("RougeL(answer, target)")
            plot.figure.savefig('plots/nlg_hist_log_rougeL_{}_{}_{}.svg'.format(dataset, key, model_name), bbox_inches='tight')
            plot.figure.clf()

            sns.set(font_scale=1.6)
            sns.set_style('ticks')
            plot = sns.kdeplot(y=result_df['rougeL_to_target'], x=np.log1p(result_df[key]+0.0001))
            plot.axes.tick_params(axis='x', rotation=45)
            sns.despine(plot.figure)
            plot.set_ylabel("RougeL(answer, target)")
            plot.figure.savefig('plots/nlg_kde_log_rougeL_{}_{}_{}.svg'.format(dataset, key, model_name), bbox_inches='tight')
            plot.figure.clf()

    wandb.finish()
    torch.cuda.empty_cache()
