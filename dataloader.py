import sys
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from scipy.misc import imresize
import pdb



class PartDataset(Dataset):
    def __init__(self, image_root='/Project0551/guoqing/scscnet/DukeMTMC/bounding_box_train/', parsing_root="/Project0551/guoqing/HPA/hppic/duke/5classes/bounding_box_train/", mode='train', transform=None):
        self.mode = mode
        self.transform = transform
        self.image_root = image_root
        self.parsing_root = parsing_root
        supported_modes = ('train', 'query', 'gallery')
        assert self.mode in supported_modes, print("Only support mode from {}".format(supported_modes))
        self.name_list = np.genfromtxt(image_root + self.mode + '_list.txt', dtype=str, delimiter=',', usecols=[0])
        self.label_list = np.genfromtxt(image_root + self.mode + '_list.txt', dtype=int, delimiter=',', usecols=[1])
        
    def __getitem__(self, index):
        img = Image.open(self.image_root + self.name_list[index])
        part_map = Image.open(self.parsing_root + self.name_list[index][:-3] + "png")

        if self.mode == 'train' and random.random() < 0.5:
            img = transforms.functional.hflip(img)
            part_map = transforms.functional.hflip(part_map)

        transforms_tensor = transforms.Compose([transforms.ToTensor()])
        img_tensor = transforms_tensor(img)
        img = self.transform(img)
        # print(type(part_map))
        # part_map = np.array(Image.fromarray(part_map).resize((96,32)))
        part_map = imresize(part_map, (96, 32), interp="nearest")
        part_map = torch.from_numpy(np.asarray(part_map, dtype=np.float))
        label = self.label_list[index]
        return img, label, part_map, part_map # img_tensor # For other purpose

    def __len__(self):
        return len(self.name_list)

