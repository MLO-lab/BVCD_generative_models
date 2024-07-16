# Kernel-based BVCD for Generative Models

This is the code that accompanies the paper ["A Bias-Variance-Covariance Decomposition of Kernel Scores for Generative Models
"](https://arxiv.org/abs/2310.05833), published at ICML 2024.

The paper contributes a bias-variance-covariance decomposition of kernel scores (kernel-based loss, similar to maximum mean discrepancy) with estimators applicable for generative models.
We perform various evaluations of generalization performance and uncertainty measures in image, audio, and language settings.
Specificaly, we discover that kernel entropy outperforms other uncertainty baselines for question answering tasks performed by large language models.

**Upper image:** Kernel entropy is used to compute the similarity of text embeddings.
**Lower image:** It outperforms other baselines for uncertainty estimation (c.f. paper for more results).

![Sketch of how kernel entropy is used for large language models](https://github.com/MLO-lab/BVCD_generative_models/blob/main/Screenshot%202024-07-15%20at%2017.34.49.png?raw=true)

## To use it in your project

To use our estimators, you can copy the file `metrics.py` in your repo.
It only requires `sklearn` and `numpy` as dependencies and should work for any recent package version.
The kernel entropy is computed via the function `kernel_noise` and the expected kernel score via `kernel_error`.

```python
import numpy as np

from metrics import dist_cov, dist_var, kernel_noise, dist_corr, kernel_error, sMMD

t1 = np.arange(0, 12).reshape(4, 3, 1)
t2 = np.arange(12, 24).reshape(4, 3, 1)
# distributional variance
var_k = dist_var(t1)
# distributional covariance
cov_k = dist_cov(t1, t2)
# distributional correlation
corr_k = dist_corr(t1, t2)
# kernel entropy
ke = kernel_noise(t1[0])
# expected kernel score
eks = kernel_error(t1, t2)
# squared MMD
smmd = sMMD(t1, t2)
```

## To reproduce our results

In this repository is all of the experimental code required to run the experiments in the corresponding paper.
Included are also the evaluation notebooks for plotting.
We first give a brief overview of the files.

### Overview

We performed experiments of three different task types (image, audio, and language generation).
We used an existing code whenever possible, but this also means that some dependency requirements are older than others.
Consequently, we include three distinct requirement files for each task type.
It is not meant to run all experiments in one go, but rather this repository should be seen as a collection of three distinct experiment structures.
We now refer in each task type the associated files, the respective requirements file, and the original code base when applicable.
We use `conda` as environment manager.

All metrics/measures are defined in `metrics.py`.
In `unit_tests.ipynb`, we performed some unit tests for these functions.

### Image Generation

The dependencies for image generation are found in `environment_image.yml`.
All image experiments can be run and plotted in `infimnist_cond_ddpm.ipynb`.
There, it is also described of how to setup infimnist.
Alternatively, one can also accelerate the computation via slurm by running `multi_ddpm.sh`, which calls `slurm_train_ddpm.sh` and `slurm_generate_samples.sh` parallelized.
Training the diffusion models is defined in `mnist__conditional_diffusion.py`.
Generating the samples for the stored model checkpoints is defined in `generate_samples.py`.

### Audio Generation

The dependencies for audio generation are found in `environment_audio.yml`.
The audio experiments are an extension of [this](https://github.com/coqui-ai/TTS/blob/dev/notebooks/Tutorial_2_train_your_first_TTS_model.ipynb) tutorial.
First, follow the tutorial instructions to setup LJSpeech.
Then, run `slurm_single_glow-tts.sh` as bash command (even though it looks like a slurm script, but it crashed for us via slurm).
It executes `glow-tts.py` for different seeds, in which the training procedure is defined.
After, run `multi_wav_generate.sh` which calls `wav_generate.sh` to generate the wav outputs for the model checkpoints.
The generations are defined in `glow-tts_generate.py`.
Plotting the results is done in `glow-tts.ipynb`.


### Natural Language Generation

The dependencies for natural language generation are found in `environment_nlg.yml`.
We mostly adopted the code of `https://github.com/lorenzkuhn/semantic_uncertainty`.
We had to do some minor adjustments due to breaking bugs, like file paths and evaluations (c.f. issues in the linked repository).
Please follow their instructions to setup the datasets.
We use `run_pipeline.sh` to run all experiments (you have to setup your own wandb account and also change the finished run ids).
We added only one new file to the pipeline, namely `get_kernel_entropy.py`, which computes the kernel entropy.
We also made a copy and extended their evaluation script in `analyze_results_kent.py` to combine all evaluations.
For the KDE plot in the appendix, run `nlg_plot_ent_scatters.py` like in `run_pipeline.sh`.
The rest of the plots can be created in the notebook `plot_nlg.ipynb`.

## Reference
If you found this work or code useful, please cite:

```
@inproceedings{gruberbias,
  title={A Bias-Variance-Covariance Decomposition of Kernel Scores for Generative Models},
  author={Gruber, Sebastian Gregor and Buettner, Florian},
  booktitle={Forty-first International Conference on Machine Learning},
  year={2024},
}
```
and possibly also the more foundational work
```
@inproceedings{gruber2023uncertainty,
  title={Uncertainty Estimates of Predictions via a General Bias-Variance Decomposition},
  author={Gruber, Sebastian Gregor and Buettner, Florian},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  pages={11331--11354},
  year={2023},
  organization={PMLR}
}
```


## License

Everything is licensed under the [MIT License](https://opensource.org/licenses/MIT).
