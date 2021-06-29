import pdb
import math
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import pdb

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU()

        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class MaxPoolingWithArgmax2D(nn.Module):
    def __init__(self, kernel_size=3, stride=2, padding=1):
        super(MaxPoolingWithArgmax2D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.maxpool_with_argmax = nn.MaxPool2d(self.kernel_size, self.stride, self.padding, return_indices=True)

    def forward(self, inputs):
        outputs, indices = self.maxpool_with_argmax(inputs)
        return outputs, indices


class MaxUnpooling2D(nn.Module):
    def __init__(self, kernel_size=3, stride=2, padding=0):
        super(MaxUnpooling2D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.max_unpool = nn.MaxUnpool2d(self.kernel_size, self.stride, self.padding)

    def forward(self, inputs, indices):
        outputs = self.max_unpool(inputs, indices)
        return outputs


class Den_Resnet(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        self.inplanes = 64
        super(Den_Resnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool_with_argmax = MaxPoolingWithArgmax2D()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=1)


        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Linear(2048, 100)



        # initialize parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, input1):
        # pdb.set_trace()
        x1 = self.conv1(input1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        # block_1_1, mask1 = self.maxpool_with_argmax(x1)
        # block_1_2, mask2 = self.maxpool_with_argmax(x2)
        # pdb.set_trace()
        block_1_1 = self.maxpool(x1)

        block_2_1 = self.layer1(block_1_1)

        block_3_1 = self.layer2(block_2_1)

        block_4_1 = self.layer3(block_3_1)

        block_5_1 = self.layer4(block_4_1)
        x_1 = self.global_avg_pool(block_5_1)
        x_1 = self.mlp(x_1.view(-1, 2048))


        return x_1


def crate_Den_Resnet_model(**kwargs):
    model = Den_Resnet(Bottleneck, [3, 4, 6, 3], **kwargs)
    checkpoint = torch.load('./model_path/resnet50_origin.pth')
    model.load_state_dict(checkpoint, strict=False)
    return model


class ConvRelu(nn.Module):
    def __init__(self, in_: int, out: int, activate=True):
        super(ConvRelu, self).__init__()
        self.activate = activate
        self.conv = nn.Conv2d(in_, out, 3, padding=1)
        # self.bn = nn.BatchNorm2d(out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.activate:
            # x = self.bn(x)
            x = self.activation(x)
        return x


