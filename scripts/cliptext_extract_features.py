import sys
sys.path.append('versatile_diffusion')
import os
import numpy as np
import pandas as pd

import torch
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型配置和加载
cfgm_name = 'vd_noema'
pth = 'versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth'
cfgm = model_cfg_bank()(cfgm_name)
net = get_model()(cfgm)
sd = torch.load(pth, map_location='cpu')
net.load_state_dict(sd, strict=False)
net.clip = net.clip.to(device)

# 从CSV加载数据
csv_path = './data/spike_stimuli/image_captions.csv'  # 替换为你的CSV文件路径
df = pd.read_csv(csv_path)

# 将前80%数据分为训练集，后20%数据分为测试集
train_split = int(0.8 * len(df))
train_data = df.iloc[:train_split]
test_data = df.iloc[train_split:]

# 提取特征函数
def extract_features(data, save_path, model, device):
    num_samples = len(data)
    num_embed = 77
    num_features = 768
    clip_features = np.zeros((num_samples, num_embed, num_features))

    with torch.no_grad():
        for i, caption in enumerate(data['caption']):  # 假设CSV中有一列名为'caption'
            if not isinstance(caption, str) or caption.strip() == "":
                print(f"Skipping empty caption at index {i}")
                continue
            
            cin = [caption]
            print(f"Processing {i + 1}/{num_samples}: {cin}")
            c = model.clip_encode_text(cin)
            clip_features[i] = c.to('cpu').numpy().mean(0)

    # 保存特征到指定路径
    np.save(save_path, clip_features)
    print(f"Features saved to {save_path}")

# 提取并保存训练集和测试集的特征
train_save_path = 'data/extracted_features/nsd_cliptext_train.npy'
test_save_path = 'data/extracted_features/nsd_cliptext_test.npy'

extract_features(train_data, train_save_path, net, device)
extract_features(test_data, test_save_path, net, device)
