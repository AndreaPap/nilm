import torch.nn as nn
from torch.nn.utils import weight_norm
import torch
        
class TCN(nn.Module):
    def __init__(self,n_inputs,n_outputs,kernel_size,stride,dilation,dropout):
        super(TCN,self).__init__()
        self.net=nn.Sequential(
            weight_norm(nn.Conv1d(n_inputs, n_outputs,kernel_size,stride=stride,padding='same',dilation=dilation)),
            nn.ReLU(),
            nn.Dropout(dropout),
            weight_norm(nn.Conv1d(n_outputs, n_outputs,kernel_size,stride=stride,padding='same',dilation=dilation)),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.downsample=nn.Conv1d(n_inputs,n_outputs,1) if n_inputs != n_outputs else None
        self.act=nn.ReLU()
        nn.init.kaiming_normal_(self.net[0].weight.data,a=0,mode='fan_in',nonlinearity='relu')
        nn.init.kaiming_normal_(self.net[3].weight.data,a=0,mode='fan_in',nonlinearity='relu')
        try:nn.init.kaiming_normal_(self.downsample.weight.data,a=0,mode='fan_in',nonlinearity='relu')
        except:None
    def forward(self,x):
        res=self.net(x)
        in_=x if self.downsample is None else self.downsample(x)
        return self.act(in_+res)
        
class TLN(nn.Module):
    def __init__(self):
        super(TLN,self).__init__()
        self.features=nn.Sequential(
            TCN(1  ,128,3,1,2**0,0.1),
            TCN(128,128,3,1,2**1,0.1),
            TCN(128,128,3,1,2**2,0.1),
            TCN(128,128,3,1,2**3,0.1),
            TCN(128,128,3,1,2**4,0.1),
            )
        self.fc1=nn.Linear(128,16)
        self.relu1=nn.ReLU()
        self.fc2=nn.Linear(16,16)
        self.relu2=nn.ReLU()
        self.fc3=nn.Linear(16,1)
        nn.init.kaiming_normal_(self.fc1.weight.data,a=0,mode='fan_in',nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight.data,a=0,mode='fan_in',nonlinearity='relu')
        #nn.init.kaiming_normal_(self.fc3.weight.data,a=0,mode='fan_in',nonlinearity='linear')
    def forward(self,x):
        y=self.features(x)#[batch,canali,dim sequenza]
        y=y.permute(0,2,1)#[batch,dim sequenza,canali]
        y=self.fc1(y)
        y=self.relu1(y)
        feat1=torch.clone(y)
        y=self.fc2(y)
        y=self.relu2(y)
        feat2=torch.clone(y)
        y=self.fc3(y)
        feat3=torch.clone(y)
        y=y.permute(0,2,1).contiguous()#[batch,canali,dim sequenza]
        feat1=feat1.permute(0,2,1).contiguous()
        feat2=feat2.permute(0,2,1).contiguous()
        feat3=feat3.permute(0,2,1).contiguous()
        return y,feat1,feat2,feat3
            
        
class Alexnet(nn.Module):
    def __init__(self):
        super(Alexnet,self).__init__()
        self.features=nn.Sequential(
            nn.Conv1d(1,128,3,stride=1,padding='same',dilation=2 ** 0),
            nn.ReLU(),
            nn.Conv1d(128,128,3,stride=1,padding='same',dilation=2 ** 1),
            nn.ReLU(),
            nn.Conv1d(128,128,3,stride=1,padding='same',dilation=2 ** 2),
            nn.ReLU(),
            nn.Conv1d(128,128,3,stride=1,padding='same',dilation=2 ** 3),
            nn.ReLU(),
            nn.Conv1d(128,128,3,stride=1,padding='same',dilation=2 ** 4),
            nn.ReLU()
            )
        self.dense=nn.Sequential(
            nn.Linear(128,16),
            nn.ReLU(),
            nn.Linear(16,16),
            nn.ReLU(),
            nn.Linear(16,1)
            )
    def forward(self,x):
        y=self.features(x)#[batch,canali,dim sequenza]
        y=y.permute(0,2,1)#[batch,dim sequenza,canali]
        y=self.dense(y)
        y=y.permute(0,2,1).contiguous()#[batch,canali,dim sequenza]
        return y,None,None,None
