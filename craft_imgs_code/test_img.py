import torch
import torch.nn as nn
import torchvision.transforms as transform

import matplotlib as plt
from PIL import Image

import myutils

from myutils import *




def red_img(img:str, net='AlexNet',use_cuda = True):
    img = Image.open(img)
    trans = transform.Compose([
        transform.Resize(input_size[net]),
        transform.ToTensor()])
    img = trans(img)
    img = img.cuda()
    return img


def eval_img(img='D:/data/VOC-Segmentation/VOCdevkit/VOC2012/JPEGImages/2007_000256.jpg'):
    img = red_img(img)
    model = myutils.load_model('AlexNet','D:/models/AlexNet/ImageNet/alexnet-1664-2768cdb3.pth')
    model.eval()
    out = model(img.unsqueeze(0))
    print(classes[torch.argmax(out).item()])

if __name__ == '__main__':
    eval_img('D:/models/imgclsmob-master/mycode/craft_images/2007_000256_30000.jpg')
    eval_img()

