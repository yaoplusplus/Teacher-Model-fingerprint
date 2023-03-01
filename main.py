# -*- coding:utf-8 -*-
# @FileName  : main.py.py
# @Time      : 2022/11/15 14:38
# @Author    : Jianyao Lyu
# @Function  : test crafted_img classification for student model
import logging
import os

import PIL.Image
import torch

from myutils import *
import finetune.load_model as load_model

logging.basicConfig(level=logging.INFO,
                    filename='log.txt',
                    format='%(asctime)s %(filename)20s %(levelname)5s | %(message)30s',
                    datefmt='%d %b %Y %H:%M:%S',
                    )


def get_test_imgs(root=craft_imgs_root):
    """
    加载 {model}/2022-11-14_21_04_57下所有图片
    :param root: root path of  crafted images
    :return: images list loaded with PIL.Image
    """
    final_root = f'{root}/2022-11-14_21_04_57'
    filenames = os.listdir(final_root)
    imgs = []
    for file in filenames:
        img = Image.open(f'{final_root}/{file}/30000.jpg')
        file = Image.open(f'{imgs_root}/{file}')
        temp = (file, img)
        imgs.append(temp)
    return imgs


def test(model, imgs):
    total = 0
    for img, craft_img in imgs:
        img = my_transforms('alexnet')(img).unsqueeze(0).to(device)
        craft_img = my_transforms('alexnet')(craft_img).unsqueeze(0).to(device)
        if (torch.argmax(model(img))).item() == (torch.argmax(model(craft_img))).item():
            total += 1
    return total / len(imgs)


def main(model, dataset):
    logging.info(f'test Teacher-printing on {model} and student model fine-tuned on {dataset}')
    # load selected samples as seed images
    pass
    # craft images for seed imgs for model
    pass
    # fine-tune model on dataset
    pass
    # make test
    loaded_models = load_model.load_finetune_model(model_name=model, dataset=dataset)
    for model in loaded_models:
        test_imgs = get_test_imgs()
        acc = test(model, test_imgs)
        logging.info(f'acc: {acc * 100:.2f}%')


if __name__ == "__main__":
    for i in range(4):
        main('alexnet', 'CIFAR10')
