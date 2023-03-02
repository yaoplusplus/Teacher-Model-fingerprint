import torch

from myutils import *

set_log_level(logging.INFO)

def basename_(model_weight_file__full_name_):
    return os.path.split(model_weight_file__full_name_)[0]


def load_finetune_model(model_name, dataset):
    """

    :param model_name:
    :param dataset:
    :return: list of loaded models
    """
    # this path exists many model weight file finetuned in different time
    model_path = f'D:/models/{model_name}/fine-tuned/ImageNet2{dataset}/2023-03-02_16_08_12'
    assert os.path.exists(model_path)
    models_dirs = os.listdir(model_path)
    logging.info(f'models: {models_dirs}')
    # 实际就是选择了三次不同时间微调的模型结果，相当于重复实验
    selected_model_dirs = random.sample(models_dirs, 3)
    logging.info(f'selected model: {selected_model_dirs}')
    models = []
    for selected_model_dir in selected_model_dirs:
        root_ = f'{model_path}/{selected_model_dir}'
        # select_model_weight_file = max(os.listdir(root_), key=basename_)
        # 因为finetune时修改了模型结构，因此保存时保存了完整的模型结构与参数，因此加载时用torch.load
        model = torch.load(root_)
        model.eval()
        models.append(model)
    return models


if __name__ == '__main__':
    models = load_finetune_model('ResNet18', 'CIFAR10')
    print(len(models))
