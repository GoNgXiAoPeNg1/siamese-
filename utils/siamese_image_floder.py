import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
import skimage.io as io
import glob
import numpy as np
import random
import pdb
import sys


def get_images(filename):
    image_names = np.genfromtxt(filename, dtype=str)
    return image_names


def load_image(file):
    return Image.open(file)


class SiameseImageTripletFloder(Dataset):
    def __init__(self, data_dir, traintxt, input_transform=None, input_transform_crop1=None):
        self.data_dir = data_dir
        self.input_transform = input_transform
        self.input_transform_crop1 = input_transform_crop1
        self.traintxt = traintxt
        self.train_names = get_images(self.traintxt)

    def __getitem__(self, index):
        name_array = self.train_names[index].split(',')

        name1 = name_array[0]
        name2 = name_array[1]
        imagename1 = self.data_dir + name1 + ".jpg"
        imagename2 = self.data_dir + name2 + ".jpg"
        with open(imagename1, "rb") as f:
            image1 = load_image(f).convert('RGB')
        with open(imagename2, "rb") as f:
            image2 = load_image(f).convert('RGB')
        if random.random() < 0.5:
            image1 = image1.transpose(Image.FLIP_LEFT_RIGHT)
            image2 = image2.transpose(Image.FLIP_LEFT_RIGHT)
        if self.input_transform is not None:
            image1 = self.input_transform(image1)
            image2 = self.input_transform(image2)
        if self.input_transform_crop1 is not None:
            image1_crop = self.input_transform_crop1(image1)
            image2_crop = self.input_transform_crop1(image2)



        return image1, image1_crop, image2, image2, image2_crop, image1

    def __len__(self):
        return len(self.train_names)


class SiameseImageTripletFloder(Dataset):
    def __init__(self, data_dir, valtxt, input_transform=None, input_transform_crop1=None):
        self.data_dir = data_dir
        self.input_transform = input_transform
        self.input_transform_crop1 = input_transform_crop1
        self.valtxt = valtxt
        self.val_names = get_images(self.valtxt)

    def __getitem__(self, index):
        name_array = self.val_names[index].split(',')

        name1 = name_array[0]
        name2 = name_array[1]
        imagename1 = self.data_dir + name1 + ".jpg"
        imagename2 = self.data_dir + name2 + ".jpg"
        with open(imagename1, "rb") as f:
            image1 = load_image(f).convert('RGB')
        with open(imagename2, "rb") as f:
            image2 = load_image(f).convert('RGB')
        if random.random() < 0.5:
            image1 = image1.transpose(Image.FLIP_LEFT_RIGHT)
            image2 = image2.transpose(Image.FLIP_LEFT_RIGHT)

        if self.input_transform is not None:
            image11 = self.input_transform(image1)
            image22 = self.input_transform(image2)
        if self.input_transform_crop1 is not None:
            image11_crop = self.input_transform_crop1(image1)
            image22_crop = self.input_transform_crop1(image2)

        return image11, image11_crop, image22, image22, image22_crop, image11

    def __len__(self):
        return len(self.val_names)
