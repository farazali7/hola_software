import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from classification.config import cfg


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def fit(self, train_loader, optimizer, epoch):
        self.train()
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = self(inputs)
            loss = nn.BCEWithLogitsLoss()(outputs, targets)
            total_loss += loss
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print('Epoch: {} {}/{} Training loss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(inputs),
                    len(train_loader.dataset),
                    loss
                ))

        print('Training loss: {:.6f}'.format(total_loss / len(train_loader)))

        return total_loss / len(train_loader)

    def evaluate(self, test_loader):
        self.eval()
        loss = 0
        correct = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = self(inputs)
                loss += nn.BCEWithLogitsLoss()(outputs, targets)
                predictions = (torch.nn.Sigmoid()(outputs) > 0.5).int()
                correct += predictions.eq(targets.view_as(predictions)).sum()

        loss = loss/len(test_loader)

        test_percent_acc = 100. * correct / len(test_loader.dataset)
        print('Test loss: {:.6f}, Test accuracy: {}/{} ({:.1f}%)\n'.format(
            loss,
            correct,
            len(test_loader.dataset),
            test_percent_acc
        ))

        return test_percent_acc, loss


class CNN_MODEL(Model):
    def __init__(self):
        super(CNN_MODEL, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=5, out_channels=32, kernel_size=5)
        self.hidden1 = nn.Linear(cfg['WINDOW_SIZE'], 512)
        self.hidden2 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.5)
        self.output = nn.Linear(256, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.flatten(x, 1)
        x = self.hidden1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = self.dropout(x)
        x = F.relu(x)

        return self.output(x)


class MLP_MODEL(Model):
    def __init__(self):
        super(MLP_MODEL, self).__init__()
        input_size = cfg['WINDOW_SIZE'] if cfg['COMBINE_CHANNELS'] else len(cfg['EMG_ELECTRODE_LOCS'])
        self.hidden1 = nn.Linear(input_size, 128)
        self.hidden2 = nn.Linear(128, 64)
        self.hidden3 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.5)
        self.output = nn.Linear(32, 1)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)

        self.apply(init_weights)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.hidden1(x)
        # x = self.dropout(x)
        x = F.relu(x)
        x = self.hidden2(x)
        # x = self.dropout(x)
        x = self.hidden3(x)
        x = F.relu(x)

        return self.output(x)
