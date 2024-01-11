"""
TODO:

MUST
[X] Write a torch.Dataset/DataLoader
     Load imagenet image as 128*128 patches. Crop 
     using torchvision.Transforms

[X] Implement Gaussian+Poisson noise during 
    transformation.

[ ] Integrate into the training loop (can be in
    other files)
""" 
import os
import os.path as osp 
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as torchvision_F
import random

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class Imagenet(Dataset):
    
    def __init__(self, opt):
        '''
        Args:
            opt (dict): A config dictionary to control loading Imagenet. The following are keys:
                path (str): Path to a directory where all sub-items are Imagenet image files
                patch_width (int) : [optional]
                patch_height (int): [optional] the desired width and height of patch to generate
                poisson_scale (float): [optional] Scaling parameter for Poisson noise
                gaussian_std (float): [optional] Standard deviation for Gaussian noise
                data_augment (callable): [optional] A callable data augmentation pipeline
        '''
        self.augment = transforms.Compose([transforms.RandomHorizontalFlip(),
                                           transforms.RandomVerticalFlip(),
                                           ])
        if "path" not in opt:
            raise ValueError(opt,": does not contain 'path' key")
        elif not osp.exists(opt["path"]):
            raise FileNotFoundError(opt["path"], " does not exist!")
        self.opt = dict(
                path=opt.get("path"),
                H=opt.get("patch_height", 128),
                W=opt.get("patch_width", 128),
                p_scale=opt.get("poisson_scale", 1.0),
                g_std=opt.get("guassian_std", 10),
                augment=opt.get("data_augment", self.augment)
                )
        self.img_list = [item.name for item in os.scandir(opt["path"]) if item.is_file and item.name.endswith("JPEG")]

    def add_noise(self, images, args={'p_scale': 1.0, 'g_std': 10.0}):
        result = []
        p_scale, g_std = args['p_scale'], args['g_std']
        if len(images.shape) <= 3: # (H, W, C) => (N, H, W, C)
            images = torch.reshape(images, (1, *images.shape))

        for n_idx in range(images.shape[0]):
            img = torch.clone(images[n_idx])
            poisson_noise = p_scale * torch.sqrt(img) * torch.randn(img.shape)
            gaussian_noise = g_std * torch.randn(img.shape)
            img = img.float() + poisson_noise + gaussian_noise
            img = torch.clamp(img, min=0, max=255)
            result.append(img.byte())
        
        return torch.cat(result)

    def __getitem__(self, index):
        """
        Return:
            noisy (torch.Tensor): noise added to clean patch
            clean (torch.Tensor): one cropped image patch (C, H, W)
        """
        if torch.is_tensor(index):
            index = index.tolist()
        clean = Image.open(osp.join(self.opt["path"], self.img_list[index]))
        # TODO 1: 确保每个image的大小都是大于 128x128
        crop = torchvision_F.crop(clean, top=random.randint(0, clean.size[0]-128), left=random.randint(0, clean.size[1]-128), height=self.opt['H'], width=self.opt['W'])
        clean = torchvision_F.to_tensor(crop)
        noisy = self.add_noise(clean, args={'p_scale': self.opt['p_scale'], 'g_std': self.opt['g_std']})
        # 给noisy, clean一块做data augment
        if self.opt['augment']:
            clean = self.opt['augment'](clean)
            noisy = self.opt['augment'](noisy)
        return noisy, clean

    def __len__(self):
        return len(self.img_list)


if __name__ == "__main__":
    imagenetset = Imagenet({"path": "/root/autodl-tmp/imagenet"})
    print("dataset length:", imagenetset.__len__())
    imagenetloader = DataLoader(imagenetset, batch_size=10, shuffle=False)
    for idx, item in enumerate(imagenetloader):
        print("Batch #", idx, ": tensor size ", item[0].size(), item[1].size())
        break

