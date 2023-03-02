import logging

from dataloader import *
from myutils import *

set_log_level(logging.INFO)
# set_log_level(logging.DEBUG)
logging.info(f'torch runs on {device}')
# 设置tensorboard记录文件保存路径
time = get_time()
SummaryWriter_path = os.path.join(tensorboard_log_directory_base, 'finetune', time)


def test(model, test_loader, test_set):
    model.eval()
    right_pred = 0
    for data, label in test_loader:
        data = data.to(device)
        label = label.to(device)
        out = model(data)
        predict = torch.argmax(out, dim=1)
        right_pred += (predict == label).sum()
    return right_pred.item() / len(test_set) * 100


# def modify_classifier(model_name, model):
#     last_linear_layer

def finetune(model_name, dataset, num_classes, lr=0.001, epochs=10, **kwargs):
    """

    :param num_classes:
    :param model_name:
    :param dataset:
    :param lr:
    :param max_iter:
    :param kwargs:
    :return:
    """
    logging.info(f'finetune {model_name} on {dataset}')
    model = getattr(models, model_name)(pretrained=True).cuda()
    assert next(model.parameters()).is_cuda
    # 修改输出层神经元个数 TODO: suited for different nets
    if model_name == 'alexnet':
        logging.debug(f'last_linear_layer: {model.classifier[6].__str__}')
        model.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes).to(device)
        fc_params = list(map(id, model.classifier[6].parameters()))
        feature_extractor_params = filter(lambda p: id(p) not in fc_params, model.parameters())

        opt = optim.Adam([{'params': feature_extractor_params, 'lr': 0},
                          {'params': model.classifier[6].parameters(), 'lr': lr}],
                         lr=lr)

    elif model_name == 'resnet18':
        logging.debug(f'last_linear_layer: {model.fc}')
        # model.fc = nn.Linear(in_features=4096, out_features=num_classes).to(device)
        model.fc = nn.Linear(in_features=512, out_features=num_classes).to(device)
        logging.debug(f'last_linear_layer after modifying: {model.fc}')
        fc_params = list(map(id, model.fc.parameters()))
        feature_extractor_params = filter(lambda p: id(p) not in fc_params, model.parameters())
        opt = optim.Adam([{'params': feature_extractor_params, 'lr': 0},
                          {'params': model.fc.parameters(), 'lr': lr}],
                         lr=lr)
    losses = []
    # 创建文件夹
    time_ = get_time()
    # save_path = f'D:/models/{model_name}/{dataset}/{time_}'
    save_path = f'D:/models/{model_name}/fine-tuned/ImageNet2{dataset}/{time_}'
    mkdirs(save_path)
    assert os.path.exists(save_path)
    train_loader, test_loader, _, test_set = getattr(DataSets, dataset)(transform=my_transforms(model_name))
    logging.info(f'{dataset} loaded, begin fine-tuning')
    model.train()
    # print(model)
    for i in range(epochs):
        for j, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = F.one_hot(y.long()).float().to(device)
            logging.debug(f'size of input image: {x.shape}')
            out = model(x)
            # print(out.size(), y.size())
            loss = nn.MSELoss()(out, y)
            if j % 100 == 0:
                logging.info(f'epochs:[{i}],iteration:[{j:3}]/[{len(train_loader)}],loss:{loss.float():.6f}')
                losses.append(loss.detach().cpu().numpy())
                plt.clf()
                plt.title("Loss")
                plt.plot(losses)
                plt.pause(0.01)
            opt.zero_grad()
            loss.backward()
            opt.step()
        acc = test(model, test_loader, test_set)
        logging.info(f'model acc: {acc:.2f}%')
        torch.save(model, f'{save_path}/{acc:.2f}.pkl')


def main():
    torch.backends.cudnn.benchmark = True
    finetune('resnet18', 'CIFAR10', num_classes=10)
    # finetune('alexnet', 'CIFAR10', num_classes=10)


if __name__ == '__main__':
    main()
