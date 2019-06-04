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
from sklearn.metrics import confusion_matrix

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

num_train = 4000
num_val = 1000
num_total = num_train + num_val
num_test = 1000
num_classes = 3

class GemLowImagesDataset(Dataset):
    def __init__(self, csv_file, root_dir, csv_file2=None, root_dir2=None, transform=None, test=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        if csv_file2 is not None:
            self.labels2 = pd.read_csv(csv_file2)
        self.root_dir2 = root_dir2
        self.transform = transform
        self.test = test

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        use_set2 = (idx >= num_total and not self.test) or (idx >= num_test + num_total and self.test)
        if use_set2:
            idx = idx - (num_total + num_test) if self.test else idx - num_total
            img_name = "well" + format(idx, '04') + "_day07_well.png"
            img_path = os.path.join(self.root_dir2, img_name)
            is_cell = self.labels2.iloc[idx][1] == 'Y'
            is_alive = num_classes == 3 and self.labels2.iloc[idx][2] == 'Y'
            if is_cell:
                label = 2 if is_alive else 1
            else:
                label = 0
        else:
            img_name = "well" + format(idx, '04') + "_day00_well.png"
            img_path = os.path.join(self.root_dir, img_name)
            label = 1 if self.labels.iloc[idx][1] == 'Y' else 0
        image = io.imread(img_path)

        if self.transform:
            image = self.transform(image)

        return (image, label)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return image[:3,:,:]


def train_model(data_loaders, model, criterion, optimizer, scheduler, num_epochs=25):
    use_gpu = torch.cuda.is_available()
    best_model_wts = model.state_dict()
    best_acc = 0.0
    dataset_sizes = {'train': num_train * 2,
                     'valid': num_val * 2}

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
                torch.save(model.state_dict(), "resnet_cnn_three_class.pth")

        print('Epoch [{}/{}] train loss: {:.4f} acc: {:.4f} '
              'valid loss: {:.4f} acc: {:.4f}'.format(
                epoch, num_epochs - 1,
                train_epoch_loss, train_epoch_acc,
                valid_epoch_loss, valid_epoch_acc))

    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
            
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        #classes = classes[unique_labels(y_true, y_pred)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')
        
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        
        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()

        plt.savefig('confusion_matrix.png')

def test_model(model, test_loader):
    test_acc = 0.0
    all_labels, all_preds = None, None
    for images, labels in test_loader:
        images, labels = images.type('torch.FloatTensor'), labels.type('torch.LongTensor')
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
        # Predict classes using images from the test set
        outputs = model(images)
        _, prediction = torch.max(outputs.data, 1)
        if all_labels is None:
            all_labels = labels.data.cpu().clone()
            all_preds = prediction.cpu().clone()
        else:
            all_labels = torch.cat((all_labels, labels.data.cpu().clone()))
            all_preds = torch.cat((all_preds, prediction.cpu().clone()))
        test_acc += torch.sum(prediction == labels.data).item()
    class_names = ["no cell", "cell"] if num_classes == 2 else ["no cell", "dead cell", "alive cell"]
    plot_confusion_matrix(all_labels, all_preds, class_names)
    test_acc = test_acc / (num_test * 2)
    print('Test acc: {:4f}'.format(test_acc))

def compute_saliency_maps(model, test_loader):
    for images, labels in test_loader:
        images, labels = images.type('torch.FloatTensor'), labels.type('torch.LongTensor')
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
        images.requires_grad_()
        saliency = None
        scores = model(images).gather(1, labels.view(-1, 1)).squeeze()
        loss = torch.sum(scores)
        loss.backward()
        saliency = images.grad.data.abs().max(dim=1)[0].cpu().numpy()
        for i in range(images.shape[0]):
            plt.imsave('saliency%d.png' % i, saliency[i], cmap=plt.cm.hot)

        
mode = 'test'
transformed_dataset = GemLowImagesDataset(csv_file='/data/day_0_labels.csv',
                                          root_dir='/data/day0_images/',
                                          csv_file2='/data/day_7_labels.csv',
                                          root_dir2='/data/day7_images/',
                                          transform=transforms.Compose([
                                               ToTensor()
                                          ]),
                                          test=mode == 'test')

if mode == 'train':
    loader_train = DataLoader(transformed_dataset, batch_size=64,
                              sampler=sampler.SubsetRandomSampler(list(range(num_train)) + list(range(num_total, num_total + num_train))))
    loader_val = DataLoader(transformed_dataset, batch_size=64,
                            sampler=sampler.SubsetRandomSampler(list(range(num_train, num_total)) + list(range(num_total + num_train, num_total * 2))))
    data_loaders = {"train": loader_train, "valid": loader_val}
    use_gpu = torch.cuda.is_available()
    print(use_gpu)
    net = models.resnet50(pretrained=False)
    # freeze all model parameters
    #for param in alexnet.parameters():
    #    param.requires_grad = False
    # new final layer with 2 classes
    num_ftrs = net.fc.in_features
    net.fc = torch.nn.Linear(num_ftrs, 3)
    #alexnet.classifier[6] = nn.Linear(4096, 2)
    if use_gpu:
        net = net.cuda()

    lr = 5e-3
    #with SummaryWriter(comment='learning rate %f' % lr) as writer:
    #    writer.add_graph(net, transformed_dataset.to_array(), verbose=True)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    #5e-4 learning rate actually leads to increase in val accuracy for alexnet
    model = train_model(data_loaders, net, criterion, optimizer, exp_lr_scheduler, num_epochs=25)
else:
    net = models.resnet50(pretrained=False)
    num_ftrs = net.fc.in_features
    net.fc = torch.nn.Linear(num_ftrs, num_classes)
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        net = net.cuda()            
    net.load_state_dict(torch.load('resnet_cnn3.pth' if num_classes != 3 else 'resnet_cnn_three_class.pth'))
    net.eval()
    loader_train = DataLoader(transformed_dataset, batch_size=64,
                              sampler=sampler.SubsetRandomSampler(list(range(num_total, num_total + num_test)) + list(range(num_total * 2 + num_test, (num_total + num_test) * 2))))
    test_model(net, loader_train)
    #compute_saliency_maps(net, loader_train)
