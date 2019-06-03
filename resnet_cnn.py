from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, sampler
import torchvision
from torchvision import transforms, datasets, models, utils
from torch.optim import lr_scheduler
from torch.autograd import Variable
from tensorboardX import SummaryWriter

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class GemLowImagesDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = "well" + format(idx, '04') + "_day00_well.png"
        img_path = os.path.join(self.root_dir, img_name)
        image = io.imread(img_path)
        label = 1 if self.labels.iloc[idx][1] == 'Y' else 0

        if self.transform:
            image = self.transform(image)

        return (image, label)

    def to_array(self):
        arr = []
        for i in range(len(self.labels)):
            arr.append(self.__getitem__(i))
        return arr

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return image[:3,:,:]

num_train = 5000
num_val = 999

def train_model(data_loaders, model, criterion, optimizer, scheduler, num_epochs=25):
    use_gpu = torch.cuda.is_available()
    best_model_wts = model.state_dict()
    best_acc = 0.0
    dataset_sizes = {'train': 5000,
                     'valid': 999}

    for epoch in range(num_epochs):
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in data_loaders[phase]:
                inputs, labels = inputs.type('torch.FloatTensor'), labels.type('torch.LongTensor')
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                #print(labels[:10], preds[:10])
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data.item()
                running_corrects += torch.sum(preds == labels.data).item()
            if phase == 'train':
                train_epoch_loss = running_loss / dataset_sizes[phase]
                train_epoch_acc = running_corrects / dataset_sizes[phase]
            else:
                valid_epoch_loss = running_loss / dataset_sizes[phase]
                valid_epoch_acc = running_corrects / dataset_sizes[phase]

            if phase == 'valid' and valid_epoch_acc > best_acc:
                best_acc = valid_epoch_acc
                best_model_wts = model.state_dict()

        print('Epoch [{}/{}] train loss: {:.4f} acc: {:.4f} '
              'valid loss: {:.4f} acc: {:.4f}'.format(
                epoch, num_epochs - 1,
                train_epoch_loss, train_epoch_acc,
                valid_epoch_loss, valid_epoch_acc))

    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model

transformed_dataset = GemLowImagesDataset(csv_file='day_0_labels.csv',
                                          root_dir='day0_images/',
                                          transform=transforms.Compose([
                                               ToTensor()
                                          ]))
loader_train = DataLoader(transformed_dataset, batch_size=64,
                          sampler=sampler.SubsetRandomSampler(range(num_train)))
loader_val = DataLoader(transformed_dataset, batch_size=64,
                        sampler=sampler.SubsetRandomSampler(range(num_train, num_train + num_val)))
data_loaders = {"train": loader_train, "valid": loader_val}

use_gpu = torch.cuda.is_available()
print(use_gpu)
net = models.resnet50(pretrained=False)
# freeze all model parameters
#for param in alexnet.parameters():
#    param.requires_grad = False

# new final layer with 2 classes
num_ftrs = net.fc.in_features
net.fc = torch.nn.Linear(num_ftrs, 2)
#alexnet.classifier[6] = nn.Linear(4096, 2)

if use_gpu:
    net = net.cuda()

lr = 5e-3
#with SummaryWriter(comment='learning rate %f' % lr) as writer:
#    writer.add_graph(net, transformed_dataset.to_array(), verbose=True)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
#5e-4 learning rate actually leads to increase in val accuracy for alexnet
model = train_model(data_loaders, net, criterion, optimizer, exp_lr_scheduler, num_epochs=25)
