# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torchvision
import copy
import pdb
from utils.siamese_image_floder import SiameseImageTripletFloder
from utils.contrastive import ContrastiveLoss
from torchvision.transforms import Compose, CenterCrop, RandomCrop, Normalize, Scale, Resize, ToTensor, ToPILImage

from model_siamese import crate_Den_Resnet_model
# --------------------路径参数--------------------

train_data_path = "/data/gongxp/codes/siamese_network-master/images/"
val_data_path = "/data/gongxp/codes/siamese_network-master/images/"
train_txt = "/data/gongxp/codes/siamese_network-master/train.txt"
val_txt = "/data/gongxp/codes/siamese_network-master/train.txt"
train_batch_size = 1
val_batch_size = 2
num_epochs = 10
GPU_id = "cuda:5"
lr_init = 0.01

# --------------------加载数据--------------------

print("Getting data...")

# transform_train = torchvision.transforms.Compose([
#     torchvision.transforms.RandomResizedCrop(224),
#     torchvision.transforms.RandomHorizontalFlip(),
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])
#
# transform_val = torchvision.transforms.Compose([
#     torchvision.transforms.Resize(256),
#     torchvision.transforms.CenterCrop(224),
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

input_transform = Compose([Resize((384, 384)), ToTensor(  # Resize((resize,resize)),
        ), Normalize([.485, .456, .406], [.229, .224, .225])])
input_transform_crop1 = Compose([RandomCrop((200,100)), ToTensor(  #
        ), Normalize([.485, .456, .406], [.229, .224, .225])])

datasets_train = SiameseImageTripletFloder(train_data_path,
                                           train_txt,
                                           input_transform=input_transform,
                                           input_transform_crop1=input_transform_crop1)

datasets_val = SiameseImageTripletFloder(val_data_path,
                                         val_txt,
                                         input_transform=input_transform,
                                         input_transform_crop1=input_transform_crop1)

dataLoader_train = torch.utils.data.DataLoader(datasets_train,
                                               batch_size=train_batch_size,
                                               shuffle=True,
                                               num_workers=2)

dataLoader_val = torch.utils.data.DataLoader(datasets_val,
                                             batch_size=val_batch_size,
                                             shuffle=True,
                                             num_workers=2)

# --------------------全局参数--------------------

device = torch.device(GPU_id if torch.cuda.is_available() else "cpu")
train_size = len(datasets_train)
val_size = len(datasets_val)

# --------------------模型--------------------


# model = torchvision.models.resnet50(pretrained=True)
model = crate_Den_Resnet_model().cuda()
# # 冻结所有层
# for param in model.parameters():
#     param.requires_grad = False


model = model.to(device)
model.train(mode=True)

# --------------------损失函数及优化算法--------------------

# criterion = ContrastiveLoss()
criterion = torch.nn.TripletMarginLoss(margin=1.0, p=2)
optimizer = torch.optim.Adam(model.parameters(), lr=lr_init, betas=(0.9, 0.99))
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

# --------------------训练--------------------

print("Training...")

# 临时保存最佳参数
best_model_wts = copy.deepcopy(model.state_dict())
best_loss = 0.0

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 20)
    # 统计训练集loss值
    running_loss = 0.0
    # 学习率衰减
    exp_lr_scheduler.step()
    # 用于打印iter轮数
    i = 0
    for inputs_1, inputs_2, inputs_3, inputs_11, inputs_22, inputs_33 in dataLoader_train:
        i += 1
        inputs_1 = inputs_1.to(device)
        inputs_2 = inputs_2.to(device)
        inputs_3 = inputs_3.to(device)

        # 将上一次迭代的梯度值置零
        optimizer.zero_grad()
        logits_1 = model(inputs_1)
        logits_2 = model(inputs_2)
        logits_3 = model(inputs_3)
        # 计算损失值
        loss = criterion(logits_1, logits_2, logits_3)
        # 反向传播
        loss.backward()
        # 更新参数
        exp_lr_scheduler.step()
        optimizer.step()
        running_loss += loss.item()
        # ----------------------------------------------------
        i += 1
        inputs_11 = inputs_11.to(device)
        inputs_22 = inputs_22.to(device)
        inputs_33 = inputs_33.to(device)

        # 将上一次迭代的梯度值置零
        optimizer.zero_grad()
        logits_1 = model(inputs_11)
        logits_2 = model(inputs_22)
        logits_3 = model(inputs_33)
        # 计算损失值
        # pdb.set_trace()
        loss = criterion(logits_1, logits_2, logits_3)
        # 反向传播
        loss.backward()
        # 更新参数
        exp_lr_scheduler.step()
        optimizer.step()
        running_loss += loss.item()
        print("Epoch : " + str(epoch) + "    Iter : " + str(i) + "    Loss : " + str(loss.item()))

    train_loss = float(running_loss) / float(i)
    print("train acc  : " + str(train_loss))

    # 统计训练集loss值
    running_loss_val = 0.0
    # 用于打印iter轮数
    j = 0
    for inputs_val_1, inputs_val_2, inputs_val_3, inputs_val_11, inputs_val_22, inputs_val_33 in dataLoader_val:
        j += 1
        inputs_val_1 = inputs_val_1.to(device)
        inputs_val_2 = inputs_val_2.to(device)
        inputs_val_3 = inputs_val_3.to(device)

        # 将上一次迭代的梯度值置零
        optimizer.zero_grad()
        logits_val_1 = model(inputs_val_1)
        logits_val_2 = model(inputs_val_2)
        logits_val_3 = model(inputs_val_3)
        # 计算损失值

        loss_val = criterion(logits_val_1, logits_val_2, logits_val_3)

        running_loss_val += loss_val.item()
        # ---------------------------------------------------
        j += 1
        inputs_val_11 = inputs_val_11.to(device)
        inputs_val_22 = inputs_val_22.to(device)
        inputs_val_33 = inputs_val_33.to(device)

        # 将上一次迭代的梯度值置零
        optimizer.zero_grad()
        logits_val_1 = model(inputs_val_11)
        logits_val_2 = model(inputs_val_22)
        logits_val_3 = model(inputs_val_33)
        # 计算损失值
        loss_val2 = criterion(logits_val_1, logits_val_2, logits_val_3)

        running_loss_val += loss_val2.item()

        loss = (loss_val.item() + loss_val2.item()) / 2.0
        print("Val Epoch : " + str(epoch) + "    Iter : " + str(j) + "    Loss : " + str(loss))

    if running_loss_val < best_loss:
        best_loss = running_loss_val
        best_model_wts = copy.deepcopy(model.state_dict())

# 加载最佳模型参数
model.load_state_dict(best_model_wts)
print(best_loss)
# 保存模型
torch.save(model.state_dict(),
           "./models/Siamese_Epoch_" + str(num_epochs) + "_loss_" + str(best_loss) + '.pkl')
