import logging
import os
import sys

import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch.optim
import get_probing_input

from myutils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.empty_cache()

set_log_level(level_=logging.INFO)


# def show(img:):

def craft_image(model, img, save_path, show_img=False, max_iter=10000, show_loss=False):
    """

    :rtype: torch.Tensor
    :param model: teacher model
    :param img: base image used to craft adv image
    :param save_path:
    :param max_iter:
    :return:
    """
    craft_img = None
    temp_imgs = []
    model.eval()
    w = torch.zeros_like(img, requires_grad=True).cuda()
    img_out = model(img)
    optimizer = optim.Adam([w], lr=0.001)
    losses = []
    for step in tqdm(range(1, max_iter + 1)):
        craft_img = 255 / 2 * (nn.Tanh()(w) + 1)
        w_out = model(craft_img)
        loss = nn.MSELoss()(img_out, w_out)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        time.sleep(0.01)
        # 显示图片并保存图片
        craft_show_img = (255 / 2 * (nn.Tanh()(torch.clone(w)) + 1))
        if step % 1000 == 0:
            # show crafted_image
            if show_loss:
                losses.append(loss.detach().cpu().item())
                plt.clf()
                plt.title("Loss")
                plt.plot(losses)
                plt.pause(0.01)
            #
            if show_img:
                show(craft_show_img)
            temp_imgs.append(craft_img)
    assert len(temp_imgs) == 10
    craft_img_label = classes_ImageNet[torch.argmax(model(craft_img)).item()]
    img_label = classes_ImageNet[torch.argmax(model(img)).item()]
    if torch.argmax(model(craft_img)).item() == torch.argmax(model(img)).item():
        os.makedirs(save_path)
        for step, craft_img in enumerate(temp_imgs):
            # PIL save method
            # craft_img = craft_img.squeeze(0)
            # craft_img = trans.ToPILImage()(craft_img)
            # craft_img.save(f'{save_path}/{step * 1000}.jpg', quality=95)
            # plt save method
            save_tensor2img(craft_img, save_path)
        return True
    else:
        logging.info(f'craft fail: {craft_img_label} and {img_label}')
        return False


def test_plt_img():
    assert os.path.exists('D:/data/VOC-Segmentation/VOCdevkit/VOC2012/JPEGImages/2007_000256.jpg')
    # img = Image.open('D:/data/VOC-Segmentation/VOCdevkit/VOC2012/JPEGImages/2007_000256.jpg')
    # print(type(img)) # <class 'PIL.JpegImagePlugin.JpegImageFile'>

    # plt.show()内嵌显示 而 Image.open()用默认照片显示器打开图片

    # img.save('test.jpg', quality=95)
    # img.save('test.jpg')
    img = plt.imread('D:/data/VOC-Segmentation/VOCdevkit/VOC2012/JPEGImages/2007_000256.jpg')
    plt.imshow(img)
    plt.show()
    plt.imsave('../test.jpg', img, pil_kwargs={'quality': 95})


def main(model_name, test=True, show_img=False):
    logging.info(f'craft img for {model_name}')
    # 使用torchvision下的权重
    model = getattr(models, model_name)(pretrained=True).cuda()
    model.eval()
    if test:
        probing_input = get_probing_input.select_input_test()
    else:
        probing_input = get_probing_input.select_input()
    logging.debug(f'probing_input: {probing_input}')
    time_ = get_time()
    craft_success_counter = 0
    for img_path in probing_input:
        logging.debug(f'img_path: {img_path}')
        img = Image.open(img_path)
        img = my_transforms(model_name)(img).unsqueeze(0).cuda()
        save_path = f'../craft_images/{model_name}/{time_}/{os.path.basename(img_path).split()[0]}'
        craft_img = craft_image(model=model, img=img, save_path=save_path, show_img=show_img, max_iter=10000)
        if craft_img:
            craft_success_counter += 1
    logging.info(f'craft images succeed: {craft_success_counter}/{len(probing_input)}')
    logging.info(f'craft succeed rate: {craft_success_counter / len(probing_input) * 100:.2f}')


# 'D:/models/AlexNet/ImageNet/alexnetb-1900-55176c6a.pth'

if __name__ == '__main__':
    main('resnet18', test=False, show_img=True)
    # test_plt_img()
