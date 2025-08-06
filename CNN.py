import torch
import torch.nn as nn


class GaussianNoise(nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = nn.Parameter(torch.tensor(std), requires_grad=False)

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x

class EmptyLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class BesicBlock(nn.Module):
    def __init__(self,chan, gausian_noise_std = 0.1):
        super(BesicBlock,self).__init__()
        self.cnn = nn.Sequential(
            GaussianNoise(gausian_noise_std),
            nn.Conv2d(chan , chan,(3,3),padding=1,bias=False),
            nn.BatchNorm2d(chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(chan, chan,(3,3),padding=1,bias=False),
            nn.BatchNorm2d(chan),
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        out = self.cnn(x)
        out = out + x
        return self.relu(out)

class TransfromBlock(nn.Module):
    def __init__(self, in_chan, out_chan, pool_size):
        super(TransfromBlock, self).__init__()
        self.cnn = nn.Sequential(
            nn.MaxPool2d(pool_size),
            nn.Conv2d(in_chan, out_chan, (1, 1), bias=False),
            nn.BatchNorm2d(out_chan),
        )
        self.conv1_1 = nn.Conv2d(in_chan, out_chan, (1,1),stride=pool_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.cnn(x)
        trn = self.conv1_1(x)
        out = out + trn
        return self.relu(out)


class MLPBlock(nn.Module):
    def __init__(self,dim, dropout = 0.1):
        super(MLPBlock,self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim * 4, dim)
        )
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        out = self.drop(x)
        out = self.MLP(out)
        out = out + x
        return self.relu(out)

class MyResCNN(nn.Module):
    def __init__(self, inpsize = (3,32,32), num_classes = 10):
        super(MyResCNN, self).__init__()
        r = [2, 2, 1, 1]
        chan = [32,64,128,256]
        stri = [2, 2, 2]
        self.cfgDict = {
            "inpsize" : inpsize,
            "num_classes" : num_classes
        }
        lis = [nn.Conv2d(3,32,kernel_size=(1,1)),
               nn.ReLU(inplace=True)]

        for i in range(len(r)):
            for j in range(r[i]):
                lis.append(BesicBlock(chan[i]))
            if i == len(r) - 1:
                lis.append(nn.Flatten())
            else :
                lis.append(TransfromBlock(in_chan=chan[i],out_chan=chan[i + 1], pool_size=stri[i]))
        self.ResCNN = nn.Sequential(*lis)
        temp = torch.zeros(inpsize).unsqueeze(0)
        temp = self.ResCNN(temp).shape[1]
        self.MLP = nn.Sequential(
            nn.Linear(temp, 768),
            MLPBlock(768),
            nn.Linear(768,num_classes)
        )
    def forward(self, inp):
        x = self.ResCNN(inp)
        return self.MLP(x)
if __name__ == "__main__":
    x = torch.randn(10,3,32,32)
    cnn = MyResCNN()
    print(cnn.forward(x).shape)
    print(cnn)