# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torchvision
import os
import PIL
from model_siamese import crate_Den_Resnet_model
from torchvision.transforms import Compose, CenterCrop, RandomCrop, Normalize, Scale, Resize, ToTensor, ToPILImage
import pdb
# --------------------参数--------------------

GPU_id = "cuda:3"
model_path = "./models/Siamese_Epoch_10_loss_0.0.pkl"
device = torch.device(GPU_id if torch.cuda.is_available() else "cpu")
root = "./test/"

# --------------------模型--------------------

print("Getting models...")

model = crate_Den_Resnet_model().cuda()
model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
model = model.to(device)
model.eval()

# --------------------测试--------------------

print("Testing...")

input_transform = Compose([Resize((384, 384)), ToTensor(  # Resize((resize,resize)),
        ), Normalize([.485, .456, .406], [.229, .224, .225])])
input_transform_crop1 = Compose([RandomCrop((200,100)), ToTensor(  #
        ), Normalize([.485, .456, .406], [.229, .224, .225])])

name_list = []

for name in os.listdir(root):
    if name != '.DS_Store':
        name_list.append(root + name)


def get_vector(filepath):
    result = []
    with open(filepath, "r") as file:
        for line in file.readlines():
            s_data = line.strip().split(",")
            s_data = [float(x) for x in s_data]
            result.append(s_data)
    return result


def dot(K, L):
    if len(K) != len(L):
        return 0

    return sum(i[0] * i[1] for i in zip(K, L))


def cal_similarity(vector1, vector2):
    num = len(vector1) 
    total_sim = dot(vector1, vector2)
    return total_sim / float(num)


for filepath in name_list:
    # 加载图像
    image_origin = PIL.Image.open(filepath)
    image = input_transform(image_origin).float()
    image_crop = input_transform_crop1(image_origin).float()

    # 升为4维，否则会报错
    image = image.unsqueeze(0)
    image_crop = image_crop.unsqueeze(0)
    image = image.to(device)
    image_crop = image_crop.to(device)
    logits = model(image)
    logits_crop = model(image_crop)

    result = logits.cpu().detach().numpy()[0]
    result_crop = logits_crop.cpu().detach().numpy()[0]
    pos_sim = cal_similarity(result, result_crop)
    print(filepath + "   pos_sim : " + str(pos_sim) )
