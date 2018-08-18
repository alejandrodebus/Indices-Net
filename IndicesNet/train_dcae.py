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
from utils import to_img
from utils import save_data

experiment = 8
directory = './R_M3E{}/dcae_img'.format(experiment)

if not os.path.exists(directory):
    os.makedirs(directory)

loss_train = []
loss_test = []

def train_DCAE(model, optimizer, epoch, device, train_loader, log_interval):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        img = data['images_lv']
        img = img.type(torch.FloatTensor).to(device)

        optimizer.zero_grad() # eliminar gradientes acumulados

        # Forward
        output,_ = model(img)
        loss = F.mse_loss(output, img)

        # Backward
        loss.backward()
        optimizer.step()

        train_loss += loss.item() # sum up batch loss

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(img), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

            pic = to_img(output.cpu().data)
            save_image(pic, './R_M3E{}/dcae_img/image_{}.png'.format(experiment, epoch))

    train_loss /= len(train_loader.dataset)
    print('\nTrain set: Average loss: {:.4f}'.format(train_loss))
    loss_train.append(train_loss)


def test_DCAE(model, device, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            img = data['images_lv']
            img = img.type(torch.FloatTensor).to(device)

            output,_ = model(img)

            test_loss += F.mse_loss(output, img, size_average=True).item() 

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
    loss_test.append(test_loss)


def main():

    torch.manual_seed(0)

    global loss_train
    global loss_test

    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

    log_interval = 4
    learning_rate_DCAE = 0.001
    num_epochs_DCAE = 600
    model_DCAE = DCAE().to(device)
    optimizer_DCAE = torch.optim.SGD(model_DCAE.parameters(), lr = learning_rate_DCAE, momentum = 0.9, weight_decay = 0.005)

    batch_size_train = 20
    batch_size_test = 20

    indices_train = np.arange(0,2020)
    indices_test = np.arange(2020,2320)
    dataset_lv = LeftVentricleDataset(indices_train, size_sand = 1)
    dataset_lv_test = LeftVentricleDataset(indices_test, size_sand = 1)
    train_loader = torch.utils.data.DataLoader(dataset = dataset_lv, batch_size = batch_size_train, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset = dataset_lv_test, batch_size = batch_size_test, **kwargs)

    for epoch in range(1, num_epochs_DCAE + 1):
        train_DCAE(model_DCAE, optimizer_DCAE, epoch, device, train_loader, log_interval)
        test_DCAE(model_DCAE, device, test_loader)

    print('Guardando datos...')
    torch.save(model_DCAE, './R_M3E{}/model_trained.pt'.format(experiment))
    save_data(loss_train, './R_M3E{}/loss_train.pickle'.format(experiment))
    save_data(loss_test, './R_M3E{}/loss_test.pickle'.format(experiment))

if __name__ == '__main__':
    main()
