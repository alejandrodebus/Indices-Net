import scipy.io as sio
import numpy as np
import torch
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os

from dataset import LeftVentricleDataset
from model_dcae import DCAE
from model_indicesnet import IndicesNet
from utils import save_data
from utils import k_folds

experiment = 1
directory = './R_Model_INET{}'.format(experiment)

if not os.path.exists(directory):
    os.makedirs(directory)

loss_train = []
loss_test = []
loss_train_fold = []
loss_test_fold = []

def train(model, optimizer, epoch, device, train_loader, log_interval):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        img = data['images_lv']
        img = img.type(torch.FloatTensor).to(device)

        label1 = data['rwt_lv']
        label1 = label1.type(torch.FloatTensor).to(device)
        label2 = data['areas_lv']
        label2 = label2.type(torch.FloatTensor).to(device)

        rwt_areas_label = torch.cat((label1, label2), 1)

        optimizer.zero_grad() # eliminar gradientes acumulados

        # Forward
        output = model(img)
        loss = F.mse_loss(output, rwt_areas_label)

        # Backward
        loss.backward()
        optimizer.step()

        train_loss += loss.item() #

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(img), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    train_loss /= len(train_loader.dataset)
    print('\nTrain set: Average loss: {:.4f}'.format(train_loss))
    loss_train.append(train_loss)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            img = data['images_lv']
            img = img.type(torch.FloatTensor).to(device)

            label1 = data['rwt_lv']
            label1 = label1.type(torch.FloatTensor).to(device)
            label2 = data['areas_lv']
            label2 = label2.type(torch.FloatTensor).to(device)

            rwt_areas_label = torch.cat((label1, label2), 1)

            output = model(img)

            test_loss += F.mse_loss(output, rwt_areas_label, size_average = False).item() # sum up batch loss

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
    loss_test.append(test_loss)


def main():

    global loss_train
    global loss_test
    global loss_train_fold
    global loss_test_fold

    torch.manual_seed(0)

    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

    batch_size_train = 20
    batch_size_test = 20

    # Load pretrained autoencoder (DCAE)
    the_model = torch.load('/home/adebus/storage/modelo3_inet/model_trained_e8.pt')
    the_model.train()

    log_interval = 20
    learning_rate = 0.001
    num_epochs = 200
    model = IndicesNet(the_model).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.9, weight_decay = 0.05)

    num_folds = 5

    for train_idx, test_idx in k_folds(n_splits = num_folds):
        print('###################### FOLD {} ######################'.format(cont_fold))
        dataset_lv_train = LeftVentricleDataset(train_idx, size_sand = 1)
        dataset_lv_test = LeftVentricleDataset(test_idx, size_sand = 1)
        train_loader = torch.utils.data.DataLoader(dataset = dataset_lv_train, batch_size = batch_size_train, **kwargs)
        test_loader = torch.utils.data.DataLoader(dataset = dataset_lv_test, batch_size = batch_size_test, **kwargs)
        loss_train = []
        loss_test = []
        for epoch in range(1, num_epochs + 1):
            train(model, optimizer,epoch, device, train_loader, log_interval)
            test(model, device, test_loader)
        cont_fold+=1
        loss_train_fold.append(loss_train)
        loss_test_fold.append(loss_test)


    print('Guardando datos...')
    torch.save(model, './R_Model_INET{}/model_trained.pt'.format(experiment))
    save_data(loss_train_fold, './R_Model_INET{}/loss_train.pickle'.format(experiment))
    save_data(loss_test_fold, './R_Model_INET{}/loss_test.pickle'.format(experiment))


if __name__ == '__main__':
    main()
