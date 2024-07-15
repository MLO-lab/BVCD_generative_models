import argparse
import torch

from mnist__conditional_diffusion import sample_per_epoch

def get_free_gpu_idx():
    """Get the index of the GPU with current lowest memory usage."""
    max_free_idx = 0
    max_free_mem = torch.cuda.mem_get_info(0)[0]
    for i in range(torch.cuda.device_count()):
        if torch.cuda.mem_get_info(i)[0] > max_free_mem:
            max_free_idx = i
            max_free_mem = torch.cuda.mem_get_info(i)[0]
    return max_free_idx

gpu_idx = get_free_gpu_idx()
device = 'cuda:{}'.format(gpu_idx)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--epoch',
    type=int,
    required=True,
)
parser.add_argument(
    '--n_sample',
    type=int,
    required=True,
    default=200
)
parser.add_argument(
    '--n_models',
    type=int,
    required=True,
    default=20
)
parser.add_argument(
    '--model_folder',
    type=str,
    required=True,
    default='models/infimnist/ddpm/setid{}_frac1.0/'
)
parser.add_argument(
    '--save_folder',
    type=str,
    required=True,
    default='models/infimnist/ddpm/generated_frac1.0/'
)
args = parser.parse_args()

torch.manual_seed(0)
sample_per_epoch(
    epochs=[args.epoch],
    ddpm_ids=range(args.n_models),
    n_sample=args.n_sample, # n_sample/10 per class
    device=device,
    save_folder=args.save_folder,
    model_file=args.model_folder+'model_ep{}.pth'
)

# logging which jobs finished
f = open("slurm_logs/generate_samples.txt", "a")
f.write("epoch {} done \n".format(args.epoch))
f.close()
