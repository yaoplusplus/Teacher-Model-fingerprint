from myutils import *

"""

generating probing input from dataset

"""


def select_input(root: str = imgs_root, baseimage_num=100):
    """select images from VOC-Segmentation dataset
    """
    imgs = []
    # TODO fix the code about select strategy
    all_imgs = os.listdir(root)
    imgs = random.sample(all_imgs, baseimage_num)
    logging.info(f'number of base images: {len(imgs)}\n')
    for i in range(len(imgs)):
        imgs[i] = os.path.join(root, imgs[i])
    return imgs


def select_input_test(root=imgs_root):
    """load a specific image for test code of crafting synthetic input"""
    return [f'{root}/2007_000256.jpg']


if __name__ == '__main__':
    set_log_level(logging.INFO)
    run_code = 0
