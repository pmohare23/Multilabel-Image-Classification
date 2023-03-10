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
        self.conv1 = nn.Conv2d(3, 64, 7,stride=2)
        
        self.conv2 = nn.Conv2d(64, 64, 1)
        self.conv3 = nn.Conv2d(64, 64, 3,padding="same")
        self.conv4 = nn.Conv2d(64, 256, 1)
        self.conv5 = nn.Conv2d(256, 64, 1)

        self.conv6 = nn.Conv2d(256, 128, 1, stride=2)
        self.conv7 = nn.Conv2d(128, 128, 3,padding="same")
        self.conv8 = nn.Conv2d(128, 512, 1)
        self.conv9 = nn.Conv2d(256, 512, 1, stride=2)
        self.conv10 = nn.Conv2d(512, 128, 1)
  
        self.conv11 = nn.Conv2d(512, 256, 1, stride=2)
        self.conv12 = nn.Conv2d(256, 256, 3,padding="same")
        self.conv13 = nn.Conv2d(256, 1024, 1)
        self.conv14 = nn.Conv2d(512, 1024, 1, stride=2)
        self.conv15 = nn.Conv2d(1024, 256, 1)

        self.conv16 = nn.Conv2d(1024, 512, 1, stride=2)
        self.conv17 = nn.Conv2d(512, 512, 3,padding="same")
        self.conv18 = nn.Conv2d(512, 2048, 1)
        self.conv19 = nn.Conv2d(1024, 2048, 1, stride=2)
        self.conv20 = nn.Conv2d(2048, 512, 1)


        self.batch1 = nn.BatchNorm2d(64)
        self.batch2 = nn.BatchNorm2d(128)
        self.batch3 = nn.BatchNorm2d(256)
        self.batch4 = nn.BatchNorm2d(512)
        self.batch5 = nn.BatchNorm2d(1024)
        self.batch6 = nn.BatchNorm2d(2048)

        self.mpool = nn.MaxPool2d(3, 2)
        self.apool = nn.AvgPool2d(2, 2)

        self.fc1 = nn.Linear(2048 * 3 * 3, NUM_CLASSES)




    def forward(self, x):
        x= nn.ZeroPad2d(3)(x)


        x = self.mpool(F.relu(self.batch1(self.conv1(x))))


        x_skip = x
        x = F.relu(self.batch1(self.conv2(x)))
        x = F.relu(self.batch1(self.conv3(x)))
        x = self.batch3(self.conv4(x))
        x_skip = self.batch3(self.conv4(x_skip))
        x = F.relu(x+x_skip)

        for i in range(2):
            x_skip = x
            x = F.relu(self.batch1(self.conv5(x)))
            x = F.relu(self.batch1(self.conv3(x)))
            x = self.batch3(self.conv4(x))
            x = F.relu(x+x_skip)


        x_skip = x
        x = F.relu(self.batch2(self.conv6(x)))
        x = F.relu(self.batch2(self.conv7(x)))
        x = self.batch4(self.conv8(x))
        x_skip = self.batch4(self.conv9(x_skip))
        x = F.relu(x+x_skip)

        for i in range(3):
            x_skip = x
            x = F.relu(self.batch2(self.conv10(x)))
            x = F.relu(self.batch2(self.conv7(x)))
            x = self.batch4(self.conv8(x))
            x = F.relu(x+x_skip)


        x_skip = x
        x = F.relu(self.batch3(self.conv11(x)))
        x = F.relu(self.batch3(self.conv12(x)))
        x = self.batch5(self.conv13(x))
        x_skip = self.batch5(self.conv14(x_skip))
        x = F.relu(x+x_skip)

        for i in range(5):
            x_skip = x
            x = F.relu(self.batch3(self.conv15(x)))
            x = F.relu(self.batch3(self.conv12(x)))
            x = self.batch5(self.conv13(x))
            x = F.relu(x+x_skip)
       

        x_skip = x
        x = F.relu(self.batch4(self.conv16(x)))
        x = F.relu(self.batch4(self.conv17(x)))
        x = self.batch6(self.conv18(x))
        x_skip = self.batch6(self.conv19(x_skip))
        x = F.relu(x+x_skip)

        for i in range(2):
            x_skip = x
            x = F.relu(self.batch4(self.conv20(x)))
            x = F.relu(self.batch4(self.conv17(x)))
            x = self.batch6(self.conv18(x))
            x = F.relu(x+x_skip)


        x = self.apool(x)
        x = x.view(x.size()[0], 2048*3*3)
        x = self.fc1(x)

        return x