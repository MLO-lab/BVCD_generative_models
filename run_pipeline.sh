#!/bin/bash

data_fraction='1.0'
n_samples=20
for model in 'opt-125m' 'opt-350m' 'opt-1.3b' 'opt-2.7b' 'opt-6.7b' 'opt-13b'
do
    run_id=`python -c "import wandb; run_id = wandb.util.generate_id(); wandb.init(project='nlg_uncertainty', id=run_id); print(run_id)"`
    
    python generate.py --num_generations_per_prompt=$n_samples --model=$model --fraction_of_data_to_use=$data_fraction --run_id=$run_id --temperature='0.5' --num_beams='1' --top_p='1.0'
    python clean_generated_strings.py  --generation_model=$model --run_id=$run_id
    python get_semantic_similarities.py --generation_model=$model --run_id=$run_id
    python get_likelihoods.py --evaluation_model=$model --generation_model=$model --run_id=$run_id
    python get_prompting_based_uncertainty.py --run_id_for_few_shot_prompt=$run_id --run_id_for_evaluation=$run_id
    python compute_confidence_measure.py --generation_model=$model --evaluation_model=$model --run_id=$run_id
    python get_kernel_entropy.py --generation_model=$model --run_id=$run_id
done

dataset='trivia_qa'
data_fraction='1.0'
n_samples=10
for model in 'opt-125m'
do
    run_id=`python -c "import wandb; run_id = wandb.util.generate_id(); wandb.init(project='nlg_uncertainty', id=run_id); print(run_id)"`
    
    python generate.py --num_generations_per_prompt=$n_samples --model=$model --fraction_of_data_to_use=$data_fraction --run_id=$run_id --temperature='0.5' --num_beams='1' --top_p='1.0' --dataset=$dataset
    python clean_generated_strings.py  --generation_model=$model --run_id=$run_id
    python get_semantic_similarities.py --generation_model=$model --run_id=$run_id
    python get_likelihoods.py --evaluation_model=$model --generation_model=$model --run_id=$run_id
    python get_prompting_based_uncertainty.py --run_id_for_few_shot_prompt=$run_id --run_id_for_evaluation=$run_id
    python compute_confidence_measure.py --generation_model=$model --evaluation_model=$model --run_id=$run_id
    python get_kernel_entropy.py --generation_model=$model --run_id=$run_id
done

### REPLACE THE FOLLOWING run ids WITH YOUR OWN!!!
# # only coqa 20 gens and trivia 10 gens (30%)
# python analyze_results_kent.py --run_ids 3a36kjqy 16xto1lu 18hihnlx 8wkscsiw 5chgzy2b 1gg4jlam vqsivevv 386igsi5 2yotsedc 3aws5stu 38nlrhxa 349oybw1
# python nlg_plot_ent_scatters.py --run_ids 3a36kjqy 16xto1lu 18hihnlx 8wkscsiw 5chgzy2b 1gg4jlam vqsivevv 386igsi5 2yotsedc 3aws5stu 38nlrhxa 349oybw1
# # only coqa 20 gens + trivia 10 gens + trivia 15 gens
# python analyze_results_kent.py --run_ids 34webe3u r6bcg0j0 q7jwqjuc 2e9bcge4 1fg1y4wi 1rj7gglg vqsivevv 386igsi5 2yotsedc 3aws5stu 38nlrhxa 349oybw1 2nejrlo2

# add more generations
# run_id=`python -c "import wandb; run_id = wandb.util.generate_id(); wandb.init(project='nlg_uncertainty', id=run_id); print(run_id)"`
# reference_run_id='1rj7gglg'
# model='opt-13b'
# data_fraction='1.0'
# n_samples=5
# python generate_extend.py --reference_run_id=$reference_run_id --num_generations_per_prompt=$n_samples --model=$model --fraction_of_data_to_use=$data_fraction --run_id=$run_id --temperature='0.5' --num_beams='1' --top_p='1.0'
# python clean_generated_strings.py  --generation_model=$model --run_id=$run_id
# python get_semantic_similarities.py --generation_model=$model --run_id=$run_id
# python get_likelihoods.py --evaluation_model=$model --generation_model=$model --run_id=$run_id
# python get_prompting_based_uncertainty.py --run_id_for_few_shot_prompt=$run_id --run_id_for_evaluation=$run_id
# python compute_confidence_measure.py --generation_model=$model --evaluation_model=$model --run_id=$run_id
# python get_kernel_entropy.py --generation_model=$model --run_id=$run_id
