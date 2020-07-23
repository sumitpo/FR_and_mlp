#!/usr/bin/env python3
# coding=utf-8

import sys
import torch
import numpy as np
from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from time import time
# import matplotlib.pyplot as plt
import os


class lsDataset(Dataset):
    def __init__(self, path, sep=','):
        self.__data = np.genfromtxt(path, delimiter=sep).astype(np.float32)

    def __getitem__(self, index):
        instance = self.__data[index, :]
        data = torch.from_numpy(instance[:-1])
        label = torch.from_numpy(np.array(instance[-1]).astype(int))
        return data, label

    def __len__(self):
        return self.__data.shape[0]


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.fc1 = nn.Linear(9, 32)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)

        self.fc2 = nn.Linear(32, 64)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.fc2.bias.data.fill_(0.01)

        self.fc3 = nn.Linear(64, 128)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        self.fc3.bias.data.fill_(0.01)

        self.fc4 = nn.Linear(128, 256)
        torch.nn.init.xavier_uniform_(self.fc4.weight)
        self.fc4.bias.data.fill_(0.01)

        self.fc5 = nn.Linear(256, 128)
        torch.nn.init.xavier_uniform_(self.fc5.weight)
        self.fc5.bias.data.fill_(0.01)

        self.fc6 = nn.Linear(128, 64)
        torch.nn.init.xavier_uniform_(self.fc6.weight)
        self.fc6.bias.data.fill_(0.01)

        self.fc7 = nn.Linear(64, 32)
        torch.nn.init.xavier_uniform_(self.fc7.weight)
        self.fc7.bias.data.fill_(0.01)

        self.fc8 = nn.Linear(32, 32)
        torch.nn.init.xavier_uniform_(self.fc8.weight)
        self.fc8.bias.data.fill_(0.01)

        self.fc9 = nn.Linear(32, 2)
        torch.nn.init.xavier_uniform_(self.fc9.weight)
        self.fc9.bias.data.fill_(0.01)



    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        return F.softmax(self.fc9(x), dim=1)


if __name__ == '__main__':
    train_dataset = lsDataset('ls_train.csv')
    test_dataset = lsDataset('ls_test.csv')
    use_gpu = torch.cuda.is_available()

    if __debug__:
        print("train size: {}, test size: {}"
              .format(len(train_dataset),
                      len(test_dataset)))

    trainloader = DataLoader(dataset=train_dataset, batch_size=16,
                             shuffle=True)
    testloader = DataLoader(dataset=test_dataset, batch_size=16,
                            shuffle=True)
    if __debug__:
        for instances, labels in trainloader:
            print("instances: {}\nlabels: {}".format(instances, labels))

    epochs = 32
    model = Classifier()
    if use_gpu:
        model = model.cuda()
    if os.path.exists('model.bak'):
        model.load_state_dict(torch.load('model.bak'))
        model.eval()
    loss_fn = nn.CrossEntropyLoss()
    if use_gpu:
        loss_fn = loss_fn.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    correct_num = []
    correct_ratio = []
    test_num = []
    test_ratio = []
    start_t = time()
    for epoch in range(epochs):
        print("epochs {}:".format(epoch))
        running_loss = 0
        correct_num.append(0)
        for instances, labels in trainloader:
            if use_gpu:
                instances = instances.cuda()
                labels = labels.cuda()
            optimizer.zero_grad()
            output = model(instances)
            prediction = torch.argmax(output, dim=1)
            loss = loss_fn(output, labels)
            correct_num[-1] += torch.eq(prediction, labels).sum()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(running_loss / len(trainloader))
        print('correct: {}/{} = {}'.format(correct_num[-1],
              len(train_dataset),
              float(correct_num[-1]) / float(len(train_dataset))))

    end_t = time()
    print("training takes {}".format(end_t-start_t))
    torch.save(model.state_dict(), 'model.bak')
    for i in correct_num:
        correct_ratio.append(float(i) / float(len(train_dataset)))
    print(range(epochs))
    print(correct_ratio)
    print(test_ratio)
    # plt.ylim(0, 1)
    # plt.plot(range(epochs), correct_ratio)
    # plt.show()
    test_num.append(0)
    start_t = time()
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for instance, labels in testloader:
        if use_gpu:
            instance = instance.cuda()
            labels = labels.cuda()
        output = model(instance)
        prediction = torch.argmax(output, dim=1)
        test_num[-1] += torch.eq(prediction, labels).sum()
        for i in range(len(instance)):
            if prediction[i] == 1 and labels[i] == 1:
                tp += 1
            if prediction[i] == 0 and labels[i] == 0:
                tn += 1
            if prediction[i] == 1 and labels[i] == 0:
                fn += 1
            if prediction[i] == 0 and labels[i] == 1:
                fp += 1
        # print('predct:{}'.format(prediction))
        # print('labels:{}'.format(labels))
    end_t = time()
    print("testing takes {}".format(end_t-start_t))
    print('correct: {}/{} = {}'.format(test_num[-1], len(test_dataset),
          float(test_num[-1]) / float(len(test_dataset))))
    print('tp: {}, fp: {}\nfn: {},tn: {}'.format(tp, fp, fn, tn))
    sys.exit()
