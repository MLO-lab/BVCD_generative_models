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

```
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

## Installation

```
```

## To reproduce our results

### Downloading the pretrained models


### Running our experiments


## Reference
If you found this work or code useful, please cite:

```
@inproceedings{gruberbias,
  title={A Bias-Variance-Covariance Decomposition of Kernel Scores for Generative Models},
  author={Gruber, Sebastian Gregor and Buettner, Florian},
  booktitle={Forty-first International Conference on Machine Learning}
}
```
and possibly also the more foundational work
```
@inproceedings{gruber2023uncertainty,
  title={Uncertainty Estimates of Predictions via a General Bias-Variance Decomposition},
  author={Gruber, Sebastian and Buettner, Florian},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  pages={11331--11354},
  year={2023},
  organization={PMLR}
}
```


## License

Everything is licensed under the [MIT License](https://opensource.org/licenses/MIT).
