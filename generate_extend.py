import argparse
import os
import pathlib
import pickle
from lib2to3.pgen2.tokenize import tokenize

import accelerate
import config
import datasets
import evaluate
import numpy as np
import torch
import tqdm
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--type_of_question', type=str)
parser.add_argument('--num_generations_per_prompt', type=int, default=5)
parser.add_argument('--fraction_of_data_to_use', type=float, default=0.9)
parser.add_argument('--model', type=str, default='opt-350m')
parser.add_argument('--reference_run_id', type=str, default='run_1')
parser.add_argument('--run_id', type=str, default='run_2')
parser.add_argument('--temperature', type=float, default='1.0')
parser.add_argument('--num_beams', type=int, default='5')
parser.add_argument('--decoding_method', type=str, default='beam_search')
parser.add_argument('--top_p', type=float, default=1.0)
parser.add_argument('--dataset', type=str, default='trivia_qa')
args = parser.parse_args()

wandb.init(project='nlg_uncertainty', id=args.reference_run_id, config=args, resume='allow')
reference_run_name = wandb.run.name
wandb.finish()

wandb.init(project='nlg_uncertainty', id=args.run_id, config=args, resume='allow')
run_name = wandb.run.name

device = 'cuda'

# Set a seed value
seed_value = 11
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os

os.environ['PYTHONHASHSEED'] = str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value
import random

random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

#Fix torch random seed
torch.manual_seed(seed_value)

os.environ["HF_DATASETS_CACHE"] = config.hf_datasets_cache

model = AutoModelForCausalLM.from_pretrained(f"facebook/{args.model}",
                                             torch_dtype=torch.float16,
                                             cache_dir=config.hf_cache_dir).cuda()

if args.model == 'opt-30b':
    accelerate.dispatch_model(model, device_map=config.device_map)

tokenizer = AutoTokenizer.from_pretrained(
    f"facebook/{args.model}", use_fast=False, cache_dir=config.hf_cache_dir
)

opt_models = ['opt-125m', 'opt-350m', 'opt-1.3b', 'opt-2.7b', 'opt-6.7b', 'opt-13b', 'opt-30b']

dataset = datasets.load_from_disk(f'{config.output_dir}/trivia_qa')

if args.fraction_of_data_to_use < 1.0:
    train_dataset = dataset.train_test_split(test_size=(1 - args.fraction_of_data_to_use), seed=seed_value)['train']
else:
    train_dataset = dataset

def encode(examples):
    return tokenizer(examples['story'] + ' Q: ' + examples['question'] + ' A:', truncation=False, padding=False)

def encode_and_format_dataset(dataset):
    dataset = dataset.map(encode, batched=False, load_from_cache_file=False)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], output_all_columns=True)
    return dataset

questions = train_dataset
dataloader = torch.utils.data.DataLoader(questions, batch_size=1)

period_token_id = tokenizer('. ')['input_ids'][1]
eos_tokens = ['Question:', ' Question:', '\n', 'Answer:', ' Answer:', 'Q:']
question_framing_ids = [[tokenizer(eos_token)['input_ids'][1]] for eos_token in eos_tokens]
squad_metric = evaluate.load("squad")
rouge = evaluate.load('rouge')
exact_match_metric = evaluate.load("exact_match")


def get_generations(model, sequences, number_of_generations):
    """For a given model, produce a number of generation """

    with torch.no_grad():
        max_length_of_generated_sequence = 256
        sequences_ext = []
        for sequence in tqdm.tqdm(sequences):
            input_ids = sequence['prompt']
            input_length = input_ids.shape[0]

            generations = torch.ones((number_of_generations, input_length + max_length_of_generated_sequence),
                                     dtype=torch.long,
                                     device=device)
            for i in range(number_of_generations):

                generation = model.generate(input_ids.unsqueeze(0),
                                            do_sample=True,
                                            num_return_sequences=1,
                                            num_beams=args.num_beams,
                                            max_length=input_length + max_length_of_generated_sequence,
                                            eos_token_id=period_token_id,
                                            temperature=args.temperature,
                                            bad_words_ids=question_framing_ids,
                                            top_p=args.top_p)
                generations[i, :generation.shape[1]] = generation

            generations = torch.reshape(generations, (-1, number_of_generations, generations.shape[-1]))
            for i in range(generations.shape[0]):
                sequence['generations'] = torch.concat([sequence['generations'], generations[i]])

                generated_texts = []
                for generation in generations[i]:
                    generated_text = tokenizer.decode(
                        generation[input_length:], skip_special_tokens=True
                    )
                    generated_text = generated_text.split('Answer: ')[-1].split('Questions:')[0]
                    generated_texts.append(generated_text)

                sequence['generated_texts'] = sequence['generated_texts'] + generated_texts
                sequences_ext.append(sequence)

    return sequences_ext

with open(f'{config.output_dir}/{reference_run_name}/{args.model}_generations.pkl', 'rb') as outfile:
    sequences = pickle.load(outfile)

sequences_ext = get_generations(model, sequences, args.num_generations_per_prompt)

pathlib.Path(f'{config.output_dir}/' + run_name).mkdir(parents=True, exist_ok=True)

with open(f'{config.output_dir}/{run_name}/{args.model}_generations.pkl', 'wb') as outfile:
    pickle.dump(sequences_ext, outfile)
