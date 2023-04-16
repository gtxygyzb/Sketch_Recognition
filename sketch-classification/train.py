import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
import time

'''
class SketchANet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 15, 3, 0),  # [64, 71, 71]
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),  # [64, 35, 35]

            nn.Conv2d(64, 128, 5, 1, 0),  # [128, 31, 31]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),  # [128, 15, 15]

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),  # [256, 15, 15]
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),  # [256, 7, 7]

            nn.Conv2d(256, 512, (7, 7), stride=1),
            nn.ReLU(),
            nn.Dropout2d(0.5),

            nn.Conv2d(512, 512, (1, 1), stride=1),
            nn.ReLU(),
            nn.Dropout2d(0.5),
        )

        self.fc = torch.nn.Linear(512, 30)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size()[0], -1)
        return self.fc(x)
'''

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        """
        定义卷积层, nn.Conv2d官方文档：https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        nn.Conv2d包含三个重要参数：
            in_channels: 输入的通道数
            out_channels: 输出的通道数
            kernel_size: 卷积核的大小
            stride: 步长，默认为1
            padding: 填充，默认为0，即不进行填充
        """
        self.classifier = nn.Sequential(
            nn.Conv2d(1, 8, 7, stride=2, padding=1),  # (1, 128, 128) -> (6, 62, 62)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (6, 62, 62) -> (6, 31, 31)

            nn.Conv2d(8, 16, 3, 1, 1),  # (6, 31, 31) -> (16, 31, 31)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (16, 30, 30) -> (16, 15, 15)

            nn.Conv2d(16, 32, 3, 1, 1),  # (16, 15, 15) -> (32, 15, 15)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (32, 15, 15) -> (32, 7, 7)
            # 当完成卷积后，使用flatten将数据展开
            # 即将tensor的shape从(batch_size, c, h, w)变成(batch_size, c*h*w)，这样才能送给全连接层

            nn.Conv2d(32, 64, 3, 1, 1),  # (32, 7, 7) -> (64, 7, 7)
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, 1),  # (64, 7, 7) -> (128, 7, 7)
            nn.ReLU(),
            # nn.Conv2d(128, 256, 3, 1, 1),  # (64, 7, 7) -> (128, 7, 7)
            # nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Flatten(),

            # 最后接全连接层。
            nn.Linear(128*7*7, 512),
            nn.ReLU(),
            nn.Linear(512, 180),
            nn.ReLU(),
            nn.Linear(180, 50)
            # 注意这里并没有调用Softmax，也不能调Softmax
            # 这是因为Softmax被包含在了CrossEntropyLoss损失函数中
            # 如果这里调用的话，就会调用两遍，最后网络啥都学不着
        )

    def forward(self, x):
        return self.classifier(x)


if __name__ == "__main__":
    labels = os.listdir("./dataset")
    print(labels)

    tfm = transforms.Compose([
        transforms.Resize([128, 128]),
        transforms.Grayscale(),
        transforms.RandomHorizontalFlip(0.2),  # 随机水平翻转图片
        transforms.RandomRotation(6),  # 随机旋转图片
        transforms.ToTensor(),
    ])

    full_set = ImageFolder("./dataset", transform=tfm)
    train_size = int(len(full_set) * 0.85)
    valid_size = int(len(full_set) - train_size)

    train_set, valid_set = torch.utils.data.random_split(full_set, [train_size, valid_size])
    """
    train_set = ImageFolder("./dataset/train", transform=tfm)
    valid_set = ImageFolder("./dataset/test", transform=tfm)
    """
    batch_size = 32
    train_loader = DataLoader(train_set, batch_size=batch_size, drop_last=True,
                              shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, drop_last=True,
                              shuffle=True, num_workers=4, pin_memory=True)

    model = CNN().cuda()

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00025)
    num_epoch = 50

    train_acc_lis = []
    valid_acc_lis = []
    train_loss_lis = []
    valid_loss_lis = []

    for epoch in range(num_epoch):
        epoch_start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0
        valid_acc = 0.0
        valid_loss = 0.0

        model.train()  # 启用 batch normalization 和 drop out
        for i, (X, Y) in enumerate(train_loader):
            X[X < 1.] = 00.
            X = 1. - X

            # plt.imshow(X[0, 0])
            # plt.show()
            # print(Y.shape)
            # print(Y[0])

            optimizer.zero_grad()
            train_pred = model(X.cuda())
            batch_loss = loss(train_pred, Y.cuda())
            batch_loss.backward()
            optimizer.step()

            _, train_pred = torch.max(train_pred, 1)
            # print(train_pred)
            train_acc += (train_pred == Y.cuda()).sum().item()
            train_loss += batch_loss.item()


        model.eval()
        mp = np.zeros((50, 50))
        with torch.no_grad():  # 被包住的代码不需要计算梯度
            for i, (X, Y) in enumerate(valid_loader):
                X[X < 1.] = 00.
                X = 1. - X

                valid_pred = model(X.cuda())
                batch_loss = loss(valid_pred, Y.cuda())
                _, valid_pred = torch.max(valid_pred, 1)

                for k in range(valid_pred.shape[0]):
                    # print(k, Y[k], valid_pred[k])
                    mp[Y[k]][valid_pred[k]] += 1
                valid_acc += (valid_pred == Y.cuda()).sum().item()
                valid_loss += batch_loss.item()

        if epoch == 5:
            plt.rcParams['font.sans-serif'] = ['SimHei']
            df = pd.DataFrame(mp, index=labels, columns=labels)
            plt.figure(figsize=(15, 15))
            sns.heatmap(df, mask=df < 1, annot=True, annot_kws={"weight": "bold"})
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.savefig("./logs/confusion728/confusion_matrix" + str(epoch) + ".jpg")
            plt.show()


        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' %
              (epoch + 1, num_epoch, time.time() - epoch_start_time,
               train_acc / train_set.__len__(),
               train_loss / train_set.__len__(),
               valid_acc / valid_set.__len__(),
               valid_loss / valid_set.__len__())
              )
        train_acc_lis.append(train_acc / train_set.__len__())
        train_loss_lis.append(train_loss / train_set.__len__())
        valid_acc_lis.append(valid_acc / valid_set.__len__())
        valid_loss_lis.append(valid_loss / valid_set.__len__())

    torch.save(model.state_dict(), './logs/mod.pth')
    print("saving model completed!")

    # Loss curve
    plt.plot(train_loss_lis)
    plt.plot(valid_loss_lis)
    plt.title('Loss')
    plt.legend(['train', 'valid'])
    plt.savefig('./logs/loss.png')
    plt.show()

    # Accuracy curve
    plt.plot(train_acc_lis)
    plt.plot(valid_acc_lis)
    plt.title('Accuracy')
    plt.legend(['train', 'valid'])
    plt.savefig('./logs/acc.png')
    plt.show()
