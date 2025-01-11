import sys
sys.path.append('versatile_diffusion')
import os
from PIL import Image
import numpy as np
import torch
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

# 模型配置和加载
cfgm_name = 'vd_noema'
pth = 'versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth'
cfgm = model_cfg_bank()(cfgm_name)
net = get_model()(cfgm)
sd = torch.load(pth, map_location='cpu')
net.load_state_dict(sd, strict=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.clip = net.clip.to(device)

# 自定义数据集
class BatchGeneratorExternalImages(Dataset):
    def __init__(self, image_list):
        self.images = image_list

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        img = T.functional.resize(img, (512, 512))
        img = T.functional.to_tensor(img).float()
        img = img * 2 - 1  # 将像素值归一化到 [-1, 1]
        return img

    def __len__(self):
        return len(self.images)

# 指定原始图片文件夹路径
image_dir = 'data/spike_stimuli/natural_scenes_dataset'

# 遍历文件夹，收集不包含 "_transform_" 的原始图片路径
all_images = []
for root, dirs, files in os.walk(image_dir):
    for fname in files:
        # 根据实际后缀调整，这里假设原始文件都是 .jpeg
        if fname.endswith('.jpeg') and '_transform_' not in fname:
            all_images.append(os.path.join(root, fname))

# 这里假设共 118 张原始图片
assert len(all_images) == 118, f"Expecting 118 images, but found {len(all_images)}."

# 划分数据集（80% 训练集，20% 测试集）
train_split = int(0.8 * len(all_images))
train_images = all_images[:train_split]
test_images = all_images[train_split:]

# 创建数据加载器
batch_size = 1
train_dataset = BatchGeneratorExternalImages(train_images)
test_dataset = BatchGeneratorExternalImages(test_images)
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 初始化存储
num_embed, num_features = 257, 768
num_train, num_test = len(train_dataset), len(test_dataset)
train_clip = np.zeros((num_train, num_embed, num_features))
test_clip = np.zeros((num_test, num_embed, num_features))

# 提取特征并保存
with torch.no_grad():
    for i, cin in enumerate(testloader):
        print(f"Processing test image {i + 1}/{num_test}")
        c = net.clip_encode_vision(cin)
        test_clip[i] = c[0].cpu().numpy()

    np.save('data/extracted_features/nsd_clipvision_test.npy', test_clip)
    print("Test features saved.")

    for i, cin in enumerate(trainloader):
        print(f"Processing train image {i + 1}/{num_train}")
        c = net.clip_encode_vision(cin)
        train_clip[i] = c[0].cpu().numpy()

    np.save('data/extracted_features/nsd_clipvision_train.npy', train_clip)
    print("Train features saved.")
