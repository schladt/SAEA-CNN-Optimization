import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


def get_loaders(data_dir, dataset='cifar', batch_size=128, train_transforms=None, test_transforms=None, download=True):
    # Get datasets
    dataset = datasets.CIFAR10 if 'cifar' in dataset.lower() else datasets.MNIST

    # Set up transforms
    if train_transforms is None:
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

    if test_transforms is None:
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])


    train_dataset = dataset(root=data_dir, train=True, download=download, transform=train_transforms)
    test_dataset = dataset(root=data_dir, train=False, download=download, transform=test_transforms)

    # Create loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


class AlexNet(nn.Module):
    # https://www.analyticsvidhya.com/blog/2021/03/introduction-to-the-architecture-of-alexnet/#:~:text=The%20Alexnet%20has%20eight%20layers,layers%20except%20the%20output%20layer.
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))

        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))

        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())

        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())

        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(9216, 4096),
            nn.ReLU())

        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())

        self.fc3 = nn.Sequential(
            nn.Linear(4096, num_classes))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


class SmallCNN(nn.Module):
    def __init__(self, num_classes=10, conv1_count=2, conv2_count=4, fc_size=128):
        super(SmallCNN, self).__init__()

        self.conv1 = torch.nn.Sequential(
            # Convolutional Layer 1
            nn.Conv2d(3, 128, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1)
        )

        self.conv2 = torch.nn.Sequential(
            # Convolutional Layer 2
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1)
        )

        self.conv3 = torch.nn.Sequential(
            # Convolutional Layer 3
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.fc = torch.nn.Sequential(
            # Classifier
            torch.nn.Flatten(),
            torch.nn.Linear(512, fc_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(fc_size, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)
        return x

    def get_num_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        return sum(np.prod(p.size()) for p in model_parameters)


def _compute_accuracy(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        correct_pred, num_examples = 0, 0
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)

            logits = model(features)
            _, predicted_labels = torch.max(logits, 1)
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100


def _compute_epoch_loss(model, data_loader, loss_fn, device):
    model.eval()
    curr_loss, num_examples = 0., 0
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            loss = loss_fn(logits, targets, reduction='sum')
            num_examples += targets.size(0)
            curr_loss += loss

        curr_loss = curr_loss / num_examples
        return curr_loss


def train_model(model, num_epochs, optimizer, device, train_loader, valid_loader=None, loss_fn=F.cross_entropy, logging_interval=1, print_=False, valid_target=None):
    log_dict = {
        'train_loss_per_batch': [],
        'train_acc_per_epoch': [],
        'train_loss_per_epoch': [],
        'valid_acc_per_epoch': [],
        'valid_loss_per_epoch': []
    }

    model = model.to(device)

    for epoch in range(num_epochs):
        # = TRAINING = #
        model.train()
        batch_num = 0
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)

            # Forward and back prop
            logits = model(features)
            loss = loss_fn(logits, targets)
            optimizer.zero_grad()
            loss.backward()

            # Update model parameters
            optimizer.step()

            # Logging
            log_dict['train_loss_per_batch'].append(loss.item())

            if batch_num % logging_interval == 0 and print_:
                print(f'Epoch: {epoch+1}/{num_epochs} | '
                      f'Batch {batch_num}/{len(train_loader)} | '
                      f'Loss: {loss.item():.4f}', end='\r')
            batch_num += 1

        log_dict['num_epochs_trained'] = epoch + 1

        # = EVALUATION = #
        model.eval()
        with torch.set_grad_enabled(False):
            log_dict['train_loss_per_epoch'].append(_compute_epoch_loss(model, train_loader, loss_fn, device).item())
            log_dict['train_acc_per_epoch'].append(_compute_accuracy(model, train_loader, device).item())
            if valid_loader:
                log_dict['valid_loss_per_epoch'].append(_compute_epoch_loss(model, valid_loader, loss_fn, device).item())
                log_dict['valid_acc_per_epoch'].append(_compute_accuracy(model, valid_loader, device).item())

        if print_:
            print(f'Epoch: {epoch+1}/{num_epochs} | '
                  f'Train Loss: {log_dict["train_loss_per_epoch"][-1]:.4f} | '
                  f'Train Acc: {log_dict["train_acc_per_epoch"][-1]:.2f}%')
            
            if valid_loader:
                print(f'Epoch: {epoch+1}/{num_epochs} | '
                      f'Valid Loss: {log_dict["valid_loss_per_epoch"][-1]:.4f} | '
                      f'Valid Acc: {log_dict["valid_acc_per_epoch"][-1]:.2f}%')
        
        if valid_target:
            if log_dict['valid_acc_per_epoch'][-1] > valid_target:
                if print_:
                    print(f'Validation accuracy target reached: {valid_target}')
                break

    return log_dict
