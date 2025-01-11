import sys
sys.path.append('vdvae')
import torch
import numpy as np
#from mpi4py import MPI
import socket
import argparse
import os
import json
import subprocess
from hps import Hyperparams, parse_args_and_update_hparams, add_vae_arguments
from utils import (logger,
                   local_mpi_rank,
                   mpi_size,
                   maybe_download,
                   mpi_rank)
from data import mkdir_p
from contextlib import contextmanager
import torch.distributed as dist
#from apex.optimizers import FusedAdam as AdamW
from vae import VAE
from torch.nn.parallel.distributed import DistributedDataParallel
from train_helpers import restore_params
from image_utils import *
from model_utils import *
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T
import pickle

import argparse

# Argument Parser
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-bs", "--bs", help="Batch Size", default=30, type=int)
parser.add_argument("-data", "--data_root", help="Path to image dataset root", default="./data/spike_stimuli/natural_scenes_dataset")
parser.add_argument("-pred", "--pred_latents", help="Path to predicted latents npy file", default="./data/extracted_features/nsd_vdvae_features_31.npz")
parser.add_argument("-out", "--output_dir", help="Output directory for reconstructed images", default="./results/vdvae/")
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-bs", "--bs",help="Batch Size",default=30)
args = parser.parse_args()
batch_size=int(args.bs)

print('Libs imported')

H = {'image_size': 64, 'image_channels': 3,'seed': 0, 'port': 29500, 'save_dir': './saved_models/test', 'data_root': './', 'desc': 'test', 'hparam_sets': 'imagenet64', 'restore_path': 'imagenet64-iter-1600000-model.th', 'restore_ema_path': 'vdvae/model/imagenet64-iter-1600000-model-ema.th', 'restore_log_path': 'imagenet64-iter-1600000-log.jsonl', 'restore_optimizer_path': 'imagenet64-iter-1600000-opt.th', 'dataset': 'imagenet64', 'ema_rate': 0.999, 'enc_blocks': '64x11,64d2,32x20,32d2,16x9,16d2,8x8,8d2,4x7,4d4,1x5', 'dec_blocks': '1x2,4m1,4x3,8m4,8x7,16m8,16x15,32m16,32x31,64m32,64x12', 'zdim': 16, 'width': 512, 'custom_width_str': '', 'bottleneck_multiple': 0.25, 'no_bias_above': 64, 'scale_encblock': False, 'test_eval': True, 'warmup_iters': 100, 'num_mixtures': 10, 'grad_clip': 220.0, 'skip_threshold': 380.0, 'lr': 0.00015, 'lr_prior': 0.00015, 'wd': 0.01, 'wd_prior': 0.0, 'num_epochs': 10000, 'n_batch': 4, 'adam_beta1': 0.9, 'adam_beta2': 0.9, 'temperature': 1.0, 'iters_per_ckpt': 25000, 'iters_per_print': 1000, 'iters_per_save': 10000, 'iters_per_images': 10000, 'epochs_per_eval': 1, 'epochs_per_probe': None, 'epochs_per_eval_save': 1, 'num_images_visualize': 8, 'num_variables_visualize': 6, 'num_temperatures_visualize': 3, 'mpi_size': 1, 'local_rank': 0, 'rank': 0, 'logdir': './saved_models/test/log'}
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
H = dotdict(H)

H, preprocess_fn = set_up_data(H)

print('Models is Loading')
ema_vae = load_vaes(H)

  
class batch_generator_external_images(Dataset):
    def __init__(self, data_path, transform=None):
        """
        Args:
            data_path (str): Path to the dataset directory containing image folders.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_path = data_path
        self.transform = transform
        self.image_paths = []

        # Collect all image file paths from the given directory
        for folder in sorted(os.listdir(self.data_path)):
            folder_path = os.path.join(self.data_path, folder)
            if os.path.isdir(folder_path):
                for img_name in sorted(os.listdir(folder_path)):
                    if img_name.endswith('.jpeg') or img_name.endswith('.jpg') or img_name.endswith('.png'):  # Add other formats if needed
                        self.image_paths.append(os.path.join(folder_path, img_name))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Get image path from the list
        img_path = self.image_paths[idx]
        
        # Open image
        img = Image.open(img_path).convert('RGB')  # Ensure image is in RGB format
        
        # Apply transform if specified
        if self.transform:
            img = self.transform(img)
        
        return img

# Define transforms and load data
transform = T.Compose([
    T.Resize((64, 64)),  # Resize to 64x64
    T.ToTensor(),        # Convert to tensor [C, H, W]
])


# Dataset Class for JPEG Images
class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        for folder in sorted(os.listdir(self.root_dir)):
            folder_path = os.path.join(self.root_dir, folder)
            if os.path.isdir(folder_path):
                for img_name in sorted(os.listdir(folder_path)):
                    if img_name.endswith('.jpeg'):
                        self.image_paths.append(os.path.join(folder_path, img_name))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

image_path = "./data/spike_stimuli/natural_scenes_dataset"
test_images = batch_generator_external_images(data_path = image_path)

# Data directory
data_root = './data/spike_stimuli/natural_scenes_dataset'
dataset = ImageFolderDataset(root_dir=data_root, transform=transform)

# Split dataset into train and test sets
split_idx = int(0.8 * len(dataset))  # 80% for train, 20% for test
train_dataset = torch.utils.data.Subset(dataset, range(0, split_idx))
test_dataset = torch.utils.data.Subset(dataset, range(split_idx, len(dataset)))

trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# testloader = DataLoader(test_images,batch_size,shuffle=False)

test_latents = []
# max_batches = 2
for i, x in enumerate(testloader):
#   if i >= max_batches:
#     break
  data_input, target = preprocess_fn(x)
  data_input = data_input.permute(0, 2, 3, 1)
  with torch.no_grad():
        print(i*batch_size)
        print(data_input.shape)
        activations = ema_vae.encoder.forward(data_input)
        px_z, stats = ema_vae.decoder.forward(activations, get_latents=True)
        #recons = ema_vae.decoder.out_net.sample(px_z)
        batch_latent = []
        for i in range(31):
            batch_latent.append(stats[i]['z'].cpu().numpy().reshape(len(data_input),-1))
        test_latents.append(np.hstack(batch_latent))
        #stats_all.append(stats)
        #imshow(imgrid(recons, cols=batch_size,pad=20))
        #imshow(imgrid(test_images[i*batch_size : (i+1)*batch_size], cols=batch_size,pad=20))
test_latents = np.concatenate(test_latents)      


pred_latents = np.load("./data/predicted_features/vdvae_predicted_latents.npz")
ref_latent = stats

# Transfor latents from flattened representation to hierarchical
def latent_transformation(latents, ref):
  layer_dims = np.array([2**4,2**4,2**8,2**8,2**8,2**8,2**10,2**10,2**10,2**10,2**10,2**10,2**10,2**10,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**14])
  transformed_latents = []
  for i in range(31):
    t_lat = latents[:,layer_dims[:i].sum():layer_dims[:i+1].sum()]
    #std_norm_test_latent = (t_lat - np.mean(t_lat,axis=0)) / np.std(t_lat,axis=0)
    #renorm_test_latent = std_norm_test_latent * np.std(kamitani_latents[i][num_test:].reshape(num_train,-1),axis=0) + np.mean(kamitani_latents[i][num_test:].reshape(num_train,-1),axis=0)
    c,h,w=ref[i]['z'].shape[1:]
    transformed_latents.append(t_lat.reshape(len(latents),c,h,w))
  return transformed_latents

idx = range(len(test_images))
print(pred_latents['predicted_latents'].shape)
input_latent = latent_transformation(pred_latents['predicted_latents'],ref_latent)

  
def sample_from_hier_latents(latents,sample_ids):
  sample_ids = [id for id in sample_ids if id<len(latents[0])]
  layers_num=len(latents)
  sample_latents = []
  for i in range(layers_num):
    sample_latents.append(torch.tensor(latents[i][sample_ids]).float().cuda())
  return sample_latents

#samples = []

for i in range(int(np.ceil(len(test_images)/batch_size))):
  print(i*batch_size)
  samp = sample_from_hier_latents(input_latent,range(i*batch_size,(i+1)*batch_size))
  px_z = ema_vae.decoder.forward_manual_latents(len(samp[0]), samp, t=None)
  sample_from_latent = ema_vae.decoder.out_net.sample(px_z)
  upsampled_images = []
  for j in range(len(sample_from_latent)):
      im = sample_from_latent[j]
      im = Image.fromarray(im)
      im = im.resize((512,512),resample=3)
      im.save('results/vdvae/{}.png'.format(i*batch_size+j))
      
