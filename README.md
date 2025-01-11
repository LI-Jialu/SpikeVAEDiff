# SpikeVAEDiff
SpikeVAEDiff: Neural Spike-based Natural Visual Scenes Reconstruction combing VDVAE and Versatile Diffusion model
## Results
The following are a few reconstructions obtained : 
<p align="center"><img src="./figures/Reconstructions.png" width="600" ></p>

## Instructions 

### Requirements
* Create `ldm` environment for running `stable diffusion v1-4` following the instructions of official GitHub repo of [stable diffusion](https://github.com/CompVis/stable-diffusion)
* `conda activate ldm` 
* Update `ldm` environment using `environment.yml` in the main directory by entering `conda env update --file environment.yml --prune` . 

### Data Acquisition and Processing
Since the `allendSDK` for acquiring spike dataset requires connection to internet outside China, but our remote server is located in China, so this part is done by `colab` separately in `data_processing.ipynb`. 
Generally speaking, the following steps are done in this `data_processing.ipynb`: 
- **spike** data download, preprocess, PCA 
    -> `design_array` 
- **stimuli** data download 
    -> put the processed image `spike_stimuli` under `./data` 
- utilizing **BLIP2** to generate caption for each image. -> put `image_captions.csv` under `./data/spike_stimuli/` 

### First Stage Reconstruction with VDVAE

1. Download pretrained VDVAE model files and put them in `vdvae/model/` folder
```
wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/imagenet64-iter-1600000-log.jsonl
wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/imagenet64-iter-1600000-model.th
wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/imagenet64-iter-1600000-model-ema.th
wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/imagenet64-iter-1600000-opt.th
```
2. Extract VDVAE latent features of stimuli images for using `python scripts/vdvae_extract_features.py`. 
3. Train regression models from spike to VDVAE latent features and save test predictions using `python scripts/vdvae_regression.py`
4. Reconstruct images from predicted test features using `python scripts/vdvae_reconstruct_images.py`

### Second Stage Reconstruction with Versatile Diffusion

1. Download pretrained Versatile Diffusion model "vd-four-flow-v1-0-fp16-deprecated.pth", "kl-f8.pth" and "optimus-vae.pth" from [HuggingFace](https://huggingface.co/shi-labs/versatile-diffusion/tree/main/pretrained_pth) and put them in `versatile_diffusion/pretrained/` folder
2. Extract CLIP-Text features of captions using `python scripts/cliptext_extract_features.py`
3. Extract CLIP-Vision features of stimuli images using `python scripts/clipvision_extract_features.py`
4. Train regression models from spike to CLIP-Text features and save test predictions using `python scripts/cliptext_regression.py`
5. Train regression models from spike data to CLIP-Vision features and save test predictions using `python scripts/clipvision_regression.py`
6. Reconstruct images from predicted test features using `python scripts/versatilediffusion_reconstruct_images.py` . 

### Quantitative Evaluation
Although results are expected to be similar, it may vary because of variations at reconstruction
1. Save test images to directory `python scripts/save_test_images.py`
2. Extract evaluation features for test images using `python scripts/eval_extract_features.py`
3. Extract evaluation features for reconstructed images of any subject using `python scripts/eval_extract_features.py`
4. Obtain quantitative metric results for each subject using`python scripts/evaluate_reconstruction.py`

## References
- We are inspired by the work: [brain-diffuser](https://github.com/ozcelikfu/brain-diffuser/tree/main)
- Codes in vdvae directory are derived from [openai/vdvae](https://github.com/openai/vdvae)
- Codes in versatile_diffusion directory are derived from earlier version of [SHI-Labs/Versatile-Diffusion](https://github.com/SHI-Labs/Versatile-Diffusion)
- Dataset used in the studies are obtained from [Natural Scenes Dataset](https://naturalscenesdataset.org/)
