from myutils import *

warnings.filterwarnings('ignore')


def MNIST(batch_size=512, transform=trans.ToTensor()):
    print('load mnist')
    print(transform)
    mnist_train = torchvision.datasets.MNIST(root='D:data', train=True,
                                             download=True, transform=transform)
    trainloader = DataLoader(mnist_train, batch_size=batch_size,
                             shuffle=False)
    mnist_test = torchvision.datasets.MNIST(root='D:data', train=False,
                                            download=True, transform=transform)
    testloader = DataLoader(mnist_test, batch_size=batch_size,
                            shuffle=False)
    return trainloader, testloader, mnist_train, mnist_test


def CIFAR10(batch_size=128, transform=trans.ToTensor()):
    # print(batch_size, transform)
    cifar_train = torchvision.datasets.CIFAR10(root='D:data',
                                               train=True,
                                               transform=transform,
                                               download=True)

    cifar_test = torchvision.datasets.CIFAR10(root='D:data',
                                              train=False,
                                              transform=transform,
                                              download=True)
    train_loader = DataLoader(dataset=cifar_train,
                              batch_size=batch_size,
                              shuffle=False)
    test_loader = DataLoader(dataset=cifar_test,
                             batch_size=batch_size,
                             shuffle=False)
    return train_loader, test_loader, cifar_train, cifar_test


def CIFAR100(batch_size=128, transform=trans.ToTensor()):
    # print(batch_size, transform)
    cifar_train = torchvision.datasets.CIFAR100(root='D:data',
                                                train=True,
                                                transform=transform,
                                                download=True)

    cifar_test = torchvision.datasets.CIFAR100(root='D:data',
                                               train=False,
                                               transform=transform,
                                               download=True)
    train_loader = DataLoader(dataset=cifar_train,
                              batch_size=batch_size,
                              shuffle=False)
    test_loader = DataLoader(dataset=cifar_test,
                             batch_size=batch_size,
                             shuffle=False)
    return train_loader, test_loader, cifar_train, cifar_test


class DataSets:
    def __init__(self):
        self.do = 'nothing'

    MNIST = MNIST
    CIFAR10 = CIFAR10
    CIFAR100 = CIFAR100
    Dogs_vs_Cats = None
    STL10 = None
    CelebA = None


if __name__ == '__main__':
    trainloader, testloader, train_set, test_set = DataSets.CIFAR100(batch_size=128)
    print(train_set.classes)
