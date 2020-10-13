"""
Author: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import torch
import torchvision.datasets as datasets
import torch.utils.data as data
from PIL import Image
from utils.mypath import MyPath
from torchvision import transforms as tf
from glob import glob


class batsnet(datasets.ImageFolder):
    def __init__(self, root=MyPath.db_root_dir('batsnet'), split='train', transform=None):
        super(batsnet, self).__init__(root=os.path.join(root, 'batsnet_%s' %(split)),
                                         transform=None)
        self.transform = transform 
        self.split = split
        self.resize = tf.Resize(256)
    
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        im_size = img.size
        img = self.resize(img)

        if self.transform is not None:
            img = self.transform(img)

        out = {'image': img, 'target': target, 'meta': {'im_size': im_size, 'index': index}}

        return out

    def get_image(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        img = self.resize(img) 
        return img

