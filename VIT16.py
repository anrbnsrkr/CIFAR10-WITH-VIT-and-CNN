import torch
import torch.nn as nn
import torch.nn.functional as F


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
            EmptyLayer() if pool_size == 1 or pool_size ==(1,1) else nn.MaxPool2d(pool_size),
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


class EncoderResCNN(nn.Module):
    def CVT_Blocks(self, x):
        B, C, H, W = x.shape
        pad_h = (self.row_size - H % self.row_size) % self.row_size
        pad_w = (self.col_size - W % self.col_size) % self.col_size
        x = F.pad(x, (0, pad_h, 0, pad_w))
        x = x.unfold(2, int(self.row_size), int(self.row_size)).unfold(3, int(self.col_size), int(self.col_size))
        x = x.permute(0, 2, 3, 1, 4, 5)
        num_row, num_col = x.shape[1 : 3]
        x = torch.flatten(x, start_dim=0, end_dim=2)
        return x, B ,num_row, num_col

    def __init__(self, outchan = 198, row_size = 8, col_size = 8):
        super(EncoderResCNN, self).__init__()
        self.outchan  = nn.Parameter(torch.tensor(outchan, dtype=torch.int32), requires_grad=False)
        self.row_size = nn.Parameter(torch.tensor(row_size, dtype=torch.int32), requires_grad=False)
        self.col_size = nn.Parameter(torch.tensor(col_size, dtype=torch.int32), requires_grad=False)
        self.GloResCNN = nn.Sequential(
            nn.Conv2d(3,32,(1,1)),
            nn.ReLU(inplace=True),
            BesicBlock(32)
        )
        r = [1, 1, 1]
        chan = [32,64,128]
        stri = [2, 2]
        lis = []
        for i in range(len(r)):
            for j in range(r[i]):
                lis.append(BesicBlock(chan[i]))
            if i != len(r) - 1:
                lis.append(TransfromBlock(in_chan=chan[i],out_chan=chan[i + 1], pool_size=stri[i]))
            else:
                lis.append(nn.Flatten())

        self.ResCNN = nn.Sequential(*lis)

        temp = torch.empty((1,3,32,32))
        temp = self.GloResCNN(temp)
        temp, B, num_row, num_col = self.CVT_Blocks(temp)
        temp = self.ResCNN(temp)
        # print(temp.shape, ", ", self.outchan)
        self.MLP = nn.Sequential(
            nn.Linear(temp.shape[-1], outchan)
        )
    def forward(self, inp):
        # print(inp.shape)
        x = self.GloResCNN(inp)
        # print(x.shape)
        x, B, num_row, num_col = self.CVT_Blocks(x)
        # print(x.shape)
        x = self.ResCNN(x)
        x = self.MLP(x)
        x = x.view(B, -1, self.outchan)
        return x, B, num_row, num_col

class DoubleConv(nn.Module):
    def __init__(self,in_chan,out_chan):
        super(DoubleConv,self).__init__()
        self.doubleConv = nn.Sequential(
            nn.Conv2d(in_chan , out_chan,(3,3),padding=1),
            nn.GELU(),
            nn.Conv2d(out_chan, out_chan,(3,3),padding=1),
            nn.BatchNorm2d(out_chan),
            nn.GELU()
        )
    def forward(self,x):
        return self.doubleConv(x)



class Embedding(nn.Module):
    def __init__(self, embedDim, numsEmbedding, dropout = 0.1):
        super().__init__()
        self.PosEmbedX = nn.Embedding(numsEmbedding, embedDim)
        self.PosEmbedY = nn.Embedding(numsEmbedding, embedDim)
        self.cls = nn.Parameter(torch.randn(1,1,embedDim))
        self.CNN = EncoderResCNN(embedDim)
    def forward(self, img):
        out, B, num_row, num_col = self.CNN(img)
        px = self.PosEmbedX(torch.arange(num_col).to(img.device)).unsqueeze(1).unsqueeze(0).expand(B, -1, num_row, -1).to(img.device)
        py = self.PosEmbedY(torch.arange(num_row).to(img.device)).unsqueeze(0).unsqueeze(0).expand(B, num_col, -1, -1).to(img.device)
        p = px + py
        out =  out + p.view(B, num_row * num_col, -1)
        cls = self.cls.expand(B, -1, -1)
        return torch.cat([cls, out],dim = 1)


class EncoderBLock(nn.Module):
    def __init__(self, embedDim, num_heads, dropout = 0.1):
        super().__init__()
        self.embedDim = int(int(embedDim / num_heads) * num_heads)
        self.Attention = nn.MultiheadAttention(self.embedDim, num_heads, dropout= dropout, batch_first= True)
        self.MLP = nn.Sequential(
            nn.Linear(self.embedDim, self.embedDim * 4),
            nn.GELU(),
            nn.Linear(self.embedDim * 4, self.embedDim)
        )
        self.Norm1 = nn.LayerNorm(self.embedDim)
        self.Norm2 = nn.LayerNorm(self.embedDim)
        self.Drop = nn.Dropout(dropout,inplace=True)
    def forward(self, inp, mask = None):
        a_o, _ = self.Attention.forward(inp,inp,inp, key_padding_mask=mask)
        l1 = self.Norm1(inp + self.Drop(a_o))
        m_o = self.MLP(l1)
        op = self.Norm2(m_o + l1)
        return op


class PredCLS(nn.Module):
    def __init__(self,embedDim : int, num_classes: int):
        super().__init__()
        self.MLP = nn.Sequential(
            nn.Linear(embedDim, embedDim),
            nn.GELU(),
            nn.LayerNorm(embedDim),
            nn.Linear(embedDim, num_classes)
        )
    def forward(self, seq):
        return self.MLP(seq[:, 0, :])


class VIT(nn.Module):
    def __init__(self, embedDim = 64, numHeads = 12, numLayers = 8, numPosEmbedding=4, num_classes = 10, dropout = 0.1):
        super().__init__()
        embedDim = int(int(embedDim / numHeads) * numHeads)
        self.cfgDict = {
                        "embedDim" : embedDim,
                        "numHeads" : numHeads,
                        'numLayers' : numLayers,
                        "numPosEmbedding" : numPosEmbedding,
                        "num_classes" : num_classes,
                        "dropout" : dropout
                        }
        self.Embed = Embedding(embedDim, numPosEmbedding)
        self.Blocks = nn.ModuleList()
        for i in range(numLayers):
            self.Blocks.append(EncoderBLock(embedDim, numHeads, dropout))
        self.CLS = PredCLS(embedDim, num_classes)

    def forward(self, img):
        out = self.Embed.forward(img)
        for block in self.Blocks:
            out = block.forward(out)
        out = self.CLS(out)
        return out

if __name__ =="__main__":
    x = torch.randn((2,3,32,32))
    model = VIT()
    y = model.forward(x)
    print(y.shape)