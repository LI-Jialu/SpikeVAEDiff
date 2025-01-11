import sys
sys.path.append('vdvae')
import torch
import numpy as np
import os
from hps import parse_args_and_update_hparams
from utils import logger
from vae import VAE
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T
import argparse

print('Libs imported')

# Argument Parser
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-bs", "--bs", help="Batch Size", default=30, type=int)
args = parser.parse_args()
batch_size = args.bs

# Hyperparameters
H = {'image_size': 64, 'image_channels': 3, 'seed': 0, 'port': 29500, 'save_dir': './saved_models/test', 'data_root': './',
     'desc': 'test', 'hparam_sets': 'imagenet64', 'restore_path': 'imagenet64-iter-1600000-model.th',
     'restore_ema_path': 'vdvae/model/imagenet64-iter-1600000-model-ema.th',
     'restore_log_path': 'imagenet64-iter-1600000-log.jsonl',
     'restore_optimizer_path': 'imagenet64-iter-1600000-opt.th', 'dataset': 'imagenet64', 'ema_rate': 0.999,
     'enc_blocks': '64x11,64d2,32x20,32d2,16x9,16d2,8x8,8d2,4x7,4d4,1x5',
     'dec_blocks': '1x2,4m1,4x3,8m4,8x7,16m8,16x15,32m16,32x31,64m32,64x12', 'zdim': 16, 'width': 512,
     'custom_width_str': '', 'bottleneck_multiple': 0.25, 'no_bias_above': 64, 'scale_encblock': False, 'test_eval': True,
     'warmup_iters': 100, 'num_mixtures': 10, 'grad_clip': 220.0, 'skip_threshold': 380.0, 'lr': 0.00015,
     'lr_prior': 0.00015, 'wd': 0.01, 'wd_prior': 0.0, 'num_epochs': 10000, 'n_batch': 4, 'adam_beta1': 0.9,
     'adam_beta2': 0.9, 'temperature': 1.0, 'iters_per_ckpt': 25000, 'iters_per_print': 1000,
     'iters_per_save': 10000, 'iters_per_images': 10000, 'epochs_per_eval': 1, 'epochs_per_probe': None,
     'epochs_per_eval_save': 1, 'num_images_visualize': 8, 'num_variables_visualize': 6, 'num_temperatures_visualize': 3,
     'mpi_size': 1, 'local_rank': 0, 'rank': 0, 'logdir': './saved_models/test/log'}

H = type("dotdict", (dict,), {"__getattr__": dict.get, "__setattr__": dict.__setitem__})(H)

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

# Define transforms and load data
transform = T.Compose([
    T.Resize((64, 64)),  # Resize to 64x64
    T.ToTensor(),        # Convert to tensor [C, H, W]
])

# Data directory
data_root = './data/spike_stimuli/natural_scenes_dataset'
dataset = ImageFolderDataset(root_dir=data_root, transform=transform)

# Split dataset into train and test sets
split_idx = int(0.8 * len(dataset))  # 80% for train, 20% for test
train_dataset = torch.utils.data.Subset(dataset, range(0, split_idx))
test_dataset = torch.utils.data.Subset(dataset, range(split_idx, len(dataset)))

trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print('Dataset Loaded')

# Model Loading
def load_vaes(H):
    print("Loading VAE model...")
    return VAE(H)

print('Models are Loading')
ema_vae = load_vaes(H)
# Feature Extraction function
def extract_latents(loader, description):
    latents = []
    print(f"Processing {description} data...")
    for i, x in enumerate(loader):
        print(f"Processing batch {i} for {description}")
        print(f"Original batch shape: {x.shape}")  # Log the original shape of the batch

        # Adjust shape to mimic what encoder expects
        data_input = x.permute(0, 2, 3, 1).contiguous()  # Mimic [batch_size, height, width, channels]

        print(f"Adjusted shape: {data_input.shape}")

        try:
            with torch.no_grad():
                activations = ema_vae.encoder.forward(data_input)  # Encoder internally permutes back
                print(f"Activations keys: {list(activations.keys())}")
                px_z, stats = ema_vae.decoder.forward(activations, get_latents=True)

                # 动态获取潜变量数量
                num_latents = len(stats)
                print(f"Decoder output stats: {num_latents} latent layers")

                batch_latent = []
                for j in range(num_latents):
                    batch_latent.append(stats[j]['z'].cpu().numpy().reshape(len(data_input), -1))
                latents.append(np.hstack(batch_latent))
        except Exception as e:
            print(f"Error during processing batch {i} for {description}: {e}")
            raise

    return np.concatenate(latents)


# Extract latents for train and test data
train_latents = extract_latents(trainloader, "train")
test_latents = extract_latents(testloader, "test")

# Save extracted features
output_dir = f"data/extracted_features/"
os.makedirs(output_dir, exist_ok=True)
np.savez(
    f"{output_dir}/nsd_vdvae_features_31l.npz",
    train_latents=train_latents,
    test_latents=test_latents
)

print(f"Saved train and test latents to {output_dir}/nsd_vdvae_features_31.npz")



