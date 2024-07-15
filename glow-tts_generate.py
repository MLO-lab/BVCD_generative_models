import argparse
import copy
import os
import numpy as np
import pathlib
import pickle
import cdpam
import torch
import torch.nn.functional as F

# BaseDatasetConfig: defines name, formatter and path of the dataset.
from TTS.tts.configs.shared_configs import BaseDatasetConfig
# GlowTTSConfig: all model related values for training, validating and testing.
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.utils.synthesis import synthesis
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.glow_tts import GlowTTS
from TTS.utils.audio import AudioProcessor
from trainer import Trainer, TrainerArgs

from pathos.multiprocessing import ProcessingPool as Pool
from tqdm import tqdm

# run only once
# !wget -O $output_path/LJSpeech-1.1.tar.bz2 https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 
# !tar -xf $output_path/LJSpeech-1.1.tar.bz2 -C $output_path

output_path = "tts_train_dir"
folders = [name for name in os.listdir(output_path) if 'run_seed' in name]

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--epoch', type=int, default=10000)
args = parser.parse_args()

# output_path = "tts_train_dir"
# if not os.path.exists(output_path):
#     os.makedirs(output_path)
    
dataset_config = BaseDatasetConfig(
    formatter="ljspeech", meta_file_train="metadata.csv", path=os.path.join(output_path, "LJSpeech-1.1/")
)

config = GlowTTSConfig(
    batch_size=32,
    eval_batch_size=16,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=100,
    text_cleaner="phoneme_cleaners",
    use_phonemes=True,
    phoneme_language="en-us",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    print_step=25,
    print_eval=False,
    mixed_precision=True,
    output_path=output_path,
    datasets=[dataset_config],
)

ap = AudioProcessor.init_from_config(config)
tokenizer, config = TTSTokenizer.init_from_config(config)

train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

embedder = cdpam.CDPAM(dev='cpu')

eval_ids = range(100)
#eval_ids = range(len(eval_samples))
reps = range(10)

def iter_func(args):
    epoch, seed = args
    folder = [target_folder for target_folder in folders if 'run_seed{}-'.format(seed) in target_folder][-1]
    target_file = '/'.join([output_path, folder, 'checkpoint_{}.pth'.format(epoch)])
    model = GlowTTS(config, ap, tokenizer, speaker_manager=None)
    model.load_checkpoint(config, target_file)
    model.eval()
    
    results = []
    for eval_id in tqdm(eval_ids, total=len(eval_ids)):
        input_text = eval_samples[eval_id]['text']
        for rep in reps:
            gen_wav = synthesis(
                model.cpu(),
                input_text,
                config,
                use_cuda=False,
                use_griffin_lim=True,
                do_trim_silence=True
            )['wav']
            torch_wav = torch.from_numpy(gen_wav).unsqueeze(0).float()
            with torch.no_grad():
                _, a1, c1 = embedder.model.base_encoder.forward(torch_wav.unsqueeze(1))
                embedding = F.normalize(a1, dim=1).detach().numpy()
            results.append({
                'audio_unique_name': eval_samples[eval_id]['audio_unique_name'],
                'gen_wav': gen_wav.tolist(),
                'gen_emb': embedding[0],
                'gen_id': rep,
                'epoch': epoch,
                'seed': seed,
            })
    return results

results = iter_func((args.epoch, args.seed))

with open(output_path + '/wav_generations_seed{}_epoch{}.pkl'.format(args.seed, args.epoch), 'wb') as outfile:
    pickle.dump(results, outfile)