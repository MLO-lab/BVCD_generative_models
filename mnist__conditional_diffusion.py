''' 
This script does conditional image generation on MNIST, using a diffusion model
This code is from 
https://github.com/TeaPearce/Conditional_Diffusion_MNIST

It is modified from,
https://github.com/cloneofsimo/minDiffusion

Diffusion model is based on DDPM,
https://arxiv.org/abs/2006.11239

The conditioning idea is taken from 'Classifier-Free Diffusion Guidance',
https://arxiv.org/abs/2207.12598

This technique also features in ImageGen 'Photorealistic Text-to-Image Diffusion Modelswith Deep Language Understanding',
https://arxiv.org/abs/2205.11487

'''

from typing import Dict, Tuple
from tqdm.autonotebook import tqdm, trange
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2 
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        '''
        process and downscale the image feature maps
        '''
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        '''
        process and upscale the image feature maps
        '''
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat = 256, n_classes=10):
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)
        self.contextembed1 = EmbedFC(n_classes, 2*n_feat)
        self.contextembed2 = EmbedFC(n_classes, 1*n_feat)

        self.up0 = nn.Sequential(
            # nn.ConvTranspose2d(6 * n_feat, 2 * n_feat, 7, 7), # when concat temb and cemb end up w 6*n_feat
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 7, 7), # otherwise just have 2*n_feat
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, c, t, context_mask):
        # x is (noisy) image, c is context label, t is timestep, 
        # context_mask says which samples to block the context on

        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        # convert context to one hot embedding
        c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)
        
        # mask out context if context_mask == 1
        context_mask = context_mask[:, None]
        context_mask = context_mask.repeat(1,self.n_classes)
        context_mask = (-1*(1-context_mask)) # need to flip 0 <-> 1
        c = c * context_mask
        
        # embed context, time step
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        # could concatenate the context embedding here instead of adaGN
        # hiddenvec = torch.cat((hiddenvec, temb1, cemb1), 1)

        up1 = self.up0(hiddenvec)
        # up2 = self.up1(up1, down2) # if want to avoid add and multiply embeddings
        up2 = self.up1(cemb1*up1+ temb1, down2)  # add and multiply embeddings
        up3 = self.up2(cemb2*up2+ temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out


def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c):
        """
        this method is used in training, so samples t and noise randomly
        """

        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        # dropout context with some probability
        context_mask = torch.bernoulli(torch.zeros_like(c)+self.drop_prob).to(self.device)
        
        # return MSE between added noise, and our predicted noise
        return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T, context_mask))

    def sample(self, n_sample, size, device, guide_w = 0.0, store_intermediate=False, leave=False):
        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with context_mask=0 and the other context_mask=1
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance

        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1), sample initial noise
        c_i = torch.arange(0,10).to(device) # context for us just cycles throught the mnist labels
        c_i = c_i.repeat(int(n_sample/c_i.shape[0]))

        # don't drop context at test time
        context_mask = torch.zeros_like(c_i).to(device)

        # double the batch
        c_i = c_i.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 1. # makes second half of batch context free

        x_i_store = [] # keep track of generated steps in case want to plot something 
        for i in trange(self.n_T, 0, -1, leave=leave, desc='Diffusion iterations'):
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample,1,1,1)

            # double batch
            x_i = x_i.repeat(2,1,1,1)
            t_is = t_is.repeat(2,1,1,1)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            # split predictions and compute weighting
            eps = self.nn_model(x_i, c_i, t_is, context_mask)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1+guide_w)*eps1 - guide_w*eps2
            x_i = x_i[:n_sample]
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            if store_intermediate and i%20==0 or i==self.n_T or i<8:
                x_i_store.append(x_i.detach().cpu().numpy())
        
        x_i_store = np.array(x_i_store)
        return x_i, x_i_store

def train_mnist(
    dataset,
    n_epoch = 20,
    start_epoch = 0,
    batch_size = 256,
    n_T = 400, # 500
    device = "cuda:0",
    n_classes = 10,
    n_feat = 128, # 128 ok, 256 better (but slower)
    lrate = 1e-4,
    save_model = False,
    save_dir = './data/diffusion_outputs10/',
    save_every=2,
    ws_test = [0.0, 0.5, 2.0], # strength of generative guidance
    leave = True,
    state_dict_file = None,
):
    p = Path(save_dir)
    p.mkdir(parents=True, exist_ok=True)

    assert n_epoch > start_epoch
    ddpm = DDPM(nn_model=ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    ddpm.to(device)
    if start_epoch>0:
        assert state_dict_file != None
        ddpm.load_state_dict(torch.load(state_dict_file, map_location=device)['model'])
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)
    if start_epoch>0:
        optim.load_state_dict(torch.load(state_dict_file, map_location=device)['opt'])

    for ep in trange(start_epoch, n_epoch, leave=leave, desc='Epochs'):
        ddpm.train()

        # linear lrate decay
        # the original implementation stopped training at 20 epochs
        # they used lrate*(1-ep/n_epoch)
        # to replicate their training, we use the scaling based on 20 epochs, even when we train longer
        if ep < 20:
            optim.param_groups[0]['lr'] = lrate*(1-ep/20)

        pbar = tqdm(dataloader, leave=leave, desc='Batches')
        loss_ema = None
        for x, c in pbar:
            optim.zero_grad()
            x = x.to(device)
            c = c.to(device)
            loss = ddpm(x, c)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()
        
        # for eval, save an image of currently generated samples (top rows)
        # followed by real images (bottom rows)
        ddpm.eval()
        with torch.no_grad():
            n_sample = 4*n_classes
            for w_i, w in enumerate(ws_test):
                x_gen, x_gen_store = ddpm.sample(n_sample, (1, 28, 28), device, guide_w=w)

                # append some real images at bottom, order by class also
                x_real = torch.Tensor(x_gen.shape).to(device)
                for k in range(n_classes):
                    for j in range(int(n_sample/n_classes)):
                        try: 
                            idx = torch.squeeze((c == k).nonzero())[j]
                        except:
                            idx = 0
                        x_real[k+(j*n_classes)] = x[idx]

                x_all = torch.cat([x_gen, x_real])
                grid = make_grid(x_all*-1 + 1, nrow=10)
                save_image(grid, save_dir + f"image_ep{ep}_w{w}.png")

        if save_model and (ep%save_every==0 or ep==n_epoch-1):
            torch.save({'model': ddpm.state_dict(), 'opt': optim.state_dict()}, save_dir + f"model_ep{ep}.pth")

def sample_ddpms(state_dict_files, n_sample, device, leave=False):
    n_class = 10
    s_per_class = int(n_sample / n_class)
    xs_gen = torch.zeros((n_class, len(state_dict_files), s_per_class, 1, 28, 28))
    progress_bar = tqdm(
        enumerate(state_dict_files), total=len(state_dict_files),
        leave=leave, desc='Models'
    )
    for i, state_dict_file in progress_bar:
        ddpm = DDPM(
            nn_model=ContextUnet(in_channels=1, n_feat=128, n_classes=10),
            betas=(1e-4, 0.02), n_T=400, device=device, drop_prob=0.1
        )
        ddpm.to(device)
        if state_dict_file is not None:
            ddpm.load_state_dict(torch.load(state_dict_file, map_location=device)['model'])
        ddpm.eval()

        # do not forget no_grad or else OOM strikes
        with torch.no_grad():
            x_gen, _ = ddpm.sample(n_sample=n_sample, size=(1, 28, 28), device=device, store_intermediate=False)
            for k in range(n_class):
                xs_gen[k, i, :, :, :, :] = x_gen[np.arange(s_per_class)*n_class+k]
    return xs_gen

def sample_per_epoch(epochs, ddpm_ids, n_sample, device, save_folder, model_file='models/infimnist/ddpm/set_id_{}/model_{}.pth'):
    p = Path(save_folder)
    p.mkdir(parents=True, exist_ok=True)

    save_file = save_folder + "xs_gen_ep{}.pth"
    for epoch in tqdm(epochs, total=len(epochs), desc='Epochs'):
        state_dict_files = [None for i in ddpm_ids] if epoch==-1 else [
            model_file.format(i, epoch) for i in ddpm_ids
        ]
        xs_gen = sample_ddpms(state_dict_files, n_sample=n_sample, device=device)
        torch.save(xs_gen, save_file.format(epoch))


if __name__ == "__main__":
    import argparse
    import numpy as np
    import os
    import torch
    
    from torch.utils.data import DataLoader, Subset
    from torchvision import models, transforms
    from torchvision.datasets import MNIST
    
    from utils import str2bool
    
    def get_free_gpu_idx():
        """Get the index of the GPU with current lowest memory usage."""
        max_free_idx = 0
        max_free_mem = torch.cuda.mem_get_info(0)[0]
        for i in range(torch.cuda.device_count()):
            if torch.cuda.mem_get_info(i)[0] > max_free_mem:
                max_free_idx = i
                max_free_mem = torch.cuda.mem_get_info(i)[0]
        return max_free_idx
    
    DEVICE = 'cuda:{}'.format(get_free_gpu_idx())

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--set_id',
        type=int,
        required=True,
    )
    parser.add_argument(
        '--n_epoch',
        type=int,
        default=100
    )
    parser.add_argument(
        '--start_epoch',
        type=int,
        default=0
    )
    parser.add_argument(
        '--zero_frac',
        type=float,
        default=1.0
    )
    parser.add_argument(
        '--model_folder',
        type=str,
        required=True,
    )
    parser.add_argument(
        "--deep_ensemble",
        type=str2bool, 
        default=False,
    )
    parser.add_argument(
        '--class_reduced',
        type=int,
        default=0
    )
    args = parser.parse_args()
    
    torch.manual_seed(args.set_id)    
    n_epoch = args.n_epoch if args.set_id < 20 else 21

    tf = transforms.Compose([transforms.ToTensor()]) # mnist is already normalised 0 to 1
    infimnist = MNIST("data/infimnist", train=True, download=True, transform=tf)
    #infimnist.data = torch.div(infimnist.data, 255.0) # scale from [0, 255] to [0, 1]
    set_size = 60000

    if not args.deep_ensemble:
        indices = np.arange(set_size*args.set_id, set_size*(args.set_id+1))
    else:
        # same train set
        indices = np.arange(set_size)
    set_dir = args.model_folder.format(args.set_id)

    # exclude class 'class_reduced'
    mask = infimnist.targets[indices].numpy()!=int(args.class_reduced)
    # re-include frac*100% of class 'class_reduced'
    n_frac = int(set_size*args.zero_frac)
    mask |= np.concatenate([np.repeat([True], n_frac), np.repeat([False], set_size-n_frac)])

    dataset = Subset(infimnist, indices[mask])
    train_mnist(
        dataset, save_dir=set_dir, n_epoch=n_epoch,
        start_epoch=args.start_epoch, save_model=True,
        device=DEVICE, ws_test=[], leave=False,
        # only used if start_epoch>0
        state_dict_file='models/infimnist/ddpm/set_id_{}/model_ep{}.pth'.format(args.set_id, args.start_epoch-1)
    )
    # logging which jobs finished
    f = open("slurm_logs/finished_runs.txt", "a")
    f.write("set {} done \n".format(args.set_id))
    f.close()
