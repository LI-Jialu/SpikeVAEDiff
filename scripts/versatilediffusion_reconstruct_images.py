import sys
sys.path.append('versatile_diffusion')
import os
import PIL
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as tvtrans
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
from lib.model_zoo.ddim_vd import DDIMSampler_VD
from lib.experiments.sd_default import color_adjust

import argparse

# Argument Parser
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-diff_str", "--diff_str", help="Diffusion Strength", default=0.75)
parser.add_argument("-mix_str", "--mix_str", help="Mixing Strength", default=0.4)
args = parser.parse_args()
strength = float(args.diff_str)
mixing = float(args.mix_str)

def regularize_image(x):
    BICUBIC = PIL.Image.Resampling.BICUBIC
    if isinstance(x, str):
        x = Image.open(x).resize([512, 512], resample=BICUBIC)
        x = tvtrans.ToTensor()(x)
    elif isinstance(x, PIL.Image.Image):
        x = x.resize([512, 512], resample=BICUBIC)
        x = tvtrans.ToTensor()(x)
    elif isinstance(x, np.ndarray):
        x = PIL.Image.fromarray(x).resize([512, 512], resample=BICUBIC)
        x = tvtrans.ToTensor()(x)
    elif isinstance(x, torch.Tensor):
        pass
    else:
        assert False, 'Unknown image type'
    assert (x.shape[1] == 512) & (x.shape[2] == 512), 'Wrong image size'
    return x

# Load the model and weights
cfgm_name = 'vd_noema'
pth = 'versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth'
cfgm = model_cfg_bank()(cfgm_name)
net = get_model()(cfgm)
sd = torch.load(pth, map_location='cpu')
net.load_state_dict(sd, strict=False)

# Set GPUs
net.clip.cuda(0)
net.autokl.cuda(0)
sampler = DDIMSampler_VD(net)

# Load predicted latents
pred_text_path = f'data/predicted_features/nsd_cliptext_predtest.npy'
pred_vision_path = f'data/predicted_features/nsd_clipvision_predtest.npy'
pred_text = np.load(pred_text_path)
pred_vision = np.load(pred_vision_path)
pred_text = torch.tensor(pred_text).half().cuda(1)
pred_vision = torch.tensor(pred_vision).half().cuda(1)

# Diffusion parameters
n_samples = 1
ddim_steps = 50
ddim_eta = 0
scale = 7.5
net.autokl.half()

# Generate images
torch.manual_seed(0)
output_dir = f'results/versatile_diffusion/'
os.makedirs(output_dir, exist_ok=True)

for im_id in range(len(pred_vision)):
    input_image_path = f'results/vdvae/{im_id}.png'
    zim = Image.open(input_image_path)
    zim = regularize_image(zim)
    zin = zim * 2 - 1
    zin = zin.unsqueeze(0).cuda(0).half()

    init_latent = net.autokl_encode(zin)
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)

    t_enc = int(strength * ddim_steps)
    device = 'cuda:0'
    z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]).to(device))

    dummy_text = ''
    utx = net.clip_encode_text(dummy_text).cuda(1).half()
    dummy_image = torch.zeros((1, 3, 224, 224)).cuda(0)
    uim = net.clip_encode_vision(dummy_image).cuda(1).half()

    z_enc = z_enc.cuda(1)
    cim = pred_vision[im_id].unsqueeze(0)
    ctx = pred_text[im_id].unsqueeze(0)

    sampler.model.model.diffusion_model.device = 'cuda:1'
    sampler.model.model.diffusion_model.half().cuda(1)

    z = sampler.decode_dc(
        x_latent=z_enc,
        first_conditioning=[uim, cim],
        second_conditioning=[utx, ctx],
        t_start=t_enc,
        unconditional_guidance_scale=scale,
        xtype='image',
        first_ctype='vision',
        second_ctype='prompt',
        mixed_ratio=(1 - mixing),
    )

    z = z.cuda(0).half()
    x = net.autokl_decode(z)

    x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
    x = [tvtrans.ToPILImage()(xi) for xi in x]

    x[0].save(os.path.join(output_dir, f'{im_id}.png'))
    print(f"[INFO] Image {im_id} saved to {output_dir}")

print("[INFO] Image generation complete.")
