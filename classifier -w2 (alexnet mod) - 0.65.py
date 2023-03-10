import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np

NUM_CLASSES = 21

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.conv1 = nn.Conv2d(3, 96, 11,stride=4)        
        self.conv2 = nn.Conv2d(96, 256, 5, padding=2)
        self.conv3 = nn.Conv2d(256, 384, 3,padding=1)
        self.conv4 = nn.Conv2d(384, 512, 3,padding=1)
        self.conv5 = nn.Conv2d(512, 384, 3, padding=1)
        self.conv6 = nn.Conv2d(384, 256, 7, padding="same")
        self.conv7 = nn.Conv2d(256, 256, 5, padding="same")
        self.conv8 = nn.Conv2d(256, 128, 3, padding="same")
        
        self.pool = nn.MaxPool2d(3, 2)

        self.batch1 = nn.BatchNorm2d(96)
        self.batch2 = nn.BatchNorm2d(384)
        self.batch3 = nn.BatchNorm2d(256)
        self.batch4 = nn.BatchNorm2d(512)
        self.batch5 = nn.BatchNorm2d(128)

        self.densebatch = nn.BatchNorm1d(1024)

        self.drop = nn.Dropout2d(p=0.1)
        self.densedrop = nn.Dropout1d(p=0.1)

        self.fc1 = nn.Linear(128 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, NUM_CLASSES)



    def forward(self, x):
        
        x = F.relu(self.batch1(self.conv1(x)))

        x = self.pool(F.relu(self.batch3(self.conv2(x))))

        x = F.relu(self.batch2(self.conv3(x)))
        x = F.relu(self.batch4(self.conv4(x)))
        x = self.drop(x)
        x = self.pool(F.relu(self.batch2(self.conv5(x))))

        

        x = F.relu(self.batch3(self.conv6(x)))
        x = F.relu(self.conv7(x))
        x = self.pool(F.relu(self.batch5(self.conv8(x))))

        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = self.densedrop(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x