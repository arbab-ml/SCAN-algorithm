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
        self.resize = tf.Resize(360)
    
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        im_size = img.size
        img = self.resize(img)

        if self.transform is not None:
          #THIS IS WHERE WE CAN SEND ARGUEMNET TO TRANSFORM, THAT'S DEFINED IN COMMON_CONFIG
          #Making a custom iterator for transforms.customs

          ##The MISTAKE IS IMG IS AGAIN OF 360X360
            for t in self.transform.transforms:
                if (str(t).find("batsnet_transformation")!=-1):
                    img = t( img, path)
                else:
                    img = t(img)
        
          



        out = {'image': img, 'target': target, 'path':path, 'meta': {'im_size': im_size, 'index': index}}

        return out

    def get_image(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        img = self.resize(img) 
        return img

