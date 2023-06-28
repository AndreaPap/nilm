import matplotlib.pyplot
import torch
from torch.nn import MSELoss
import numpy
import random
import copy
from pytorch_adapt.layers import MMDLoss
from pytorch_adapt.layers import CORALLoss

mse=MSELoss()
mmd=MMDLoss()#dimensioni (batch,emb_size) 
coral=CORALLoss()

if(torch.cuda.is_available()==True):device=torch.device('cuda:0')
else:device=torch.device('cpu')
##########################################################################################################################################################################
#dagli array di in e out genera batch random riferiti agli stessi indici
def batchShuffle(x,y,batchSize=256,windowSize=600,windowStride=8):
    xTensor=numpy.ndarray([batchSize,1,windowSize])
    yTensor=numpy.ndarray([batchSize,1,windowSize])
    for curBatch in range(0,batchSize):
            index=random.randint(0,len(x)-windowSize-1)
            index-=index%windowStride#tutte le posizioni sono multiplo dello stride(sliding window)
            for curWindow in range(0,windowSize):
                    xTensor[curBatch][0][curWindow]=copy.deepcopy(x[index+curWindow])
                    yTensor[curBatch][0][curWindow]=copy.deepcopy(y[index+curWindow])
    xTensor=torch.tensor(xTensor,device=device,requires_grad=True,dtype=torch.float32)
    yTensor=torch.tensor(yTensor,device=device,requires_grad=True,dtype=torch.float32)
    return xTensor,yTensor
##########################################################################################################################################################################à
def overlapPlot(tracks,labels,dataPath,title):
    scale=5
    matplotlib.pyplot.rcParams.update({'font.size': 10*scale})
    figure,axis=matplotlib.pyplot.subplots(1,1,figsize=(6.4*scale, 4.8*scale),dpi=300)
    axis.set_xlabel('time (samples)')
    axis.set_ylabel('power (watt)')
    for cur in range(0,len(tracks)):
        axis.plot(tracks[cur].flatten().tolist(),label=labels[cur],linewidth=2*scale)
    matplotlib.pyplot.legend()
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.savefig(dataPath+title)
    matplotlib.pyplot.close()
################################################################################
def createSet(aggregateSourcePaths,groundTruthSourcePaths,aggregateTestingPaths,groundTruthTestingPaths,val=0.2):
    aggregateTraining=numpy.array([])
    groundTruthTraining=numpy.array([])
    aggregateValidation=numpy.array([])
    groundTruthValidation=numpy.array([])
    aggregateTesting=numpy.array([])
    groundTruthTesting=numpy.array([])
    for path in aggregateSourcePaths:
        newAggregate=numpy.load(path)
        splitIndex=int(newAggregate.shape[0]*(1-val))#divisione train e validation per ogni porzione
        aggregateTraining=numpy.concatenate([aggregateTraining,newAggregate[0:splitIndex]])
        aggregateValidation=numpy.concatenate([aggregateValidation,newAggregate[splitIndex:]])
    for path in groundTruthSourcePaths:
        newGroundTruth=numpy.load(path)
        splitIndex=int(newGroundTruth.shape[0]*(1-val))
        groundTruthTraining=numpy.concatenate([groundTruthTraining,newGroundTruth[0:splitIndex]])
        groundTruthValidation=numpy.concatenate([groundTruthValidation,newGroundTruth[splitIndex:]])
    for path in aggregateTestingPaths:
        aggregateTesting=numpy.concatenate([aggregateTesting,numpy.load(path)])
    for path in groundTruthTestingPaths:
        groundTruthTesting=numpy.concatenate([groundTruthTesting,numpy.load(path)])
    return aggregateTraining,groundTruthTraining,aggregateValidation,groundTruthValidation,aggregateTesting,groundTruthTesting
#######################################################################################
def lossFunc(adapt,groundTruth_0,pred_0,feat1_0,feat2_0,feat3_0,pred_1,feat1_1,feat2_1,feat3_1):
    #_0 dominio su cui calcolo loss normale, _1 dominio rispetto al quale adatto
    u=0.4
    l=0.6
    if(adapt==False):
        return mse(pred_0,groundTruth_0)
    if(adapt==True):
        feat1_0=feat1_0.flatten(1)
        feat2_0=feat2_0.flatten(1)
        feat3_0=feat3_0.flatten(1)
        feat1_1=feat1_1.flatten(1)
        feat2_1=feat2_1.flatten(1)
        feat3_1=feat3_1.flatten(1)
        regLoss=mse(pred_0,groundTruth_0)
        adLoss=u*mmd(feat1_0,feat1_1).pow(2)
        adLoss+=(1-u)*coral(feat1_0,feat1_1)
        adLoss+=u*mmd(feat2_0,feat2_1).pow(2)
        adLoss+=(1-u)*coral(feat2_0,feat2_1)
        adLoss+=u*mmd(feat3_0,feat3_1).pow(2)
        adLoss+=(1-u)*coral(feat3_0,feat3_1)
        #print('mse loss:',round(regLoss.item(),3),'adaptive loss:',round(adLoss.item(),3))
        return ((1-l)*regLoss)+(l*adLoss)
####################################################################################### 
def train(net,optimizer,aggregate_0,groundTruth_0,aggregate_1,groundTruth_1,adapt):
    net=net.train()
    aggregateTensor_0,groundTruthTensor_0=batchShuffle(aggregate_0,groundTruth_0)
    aggregateTensor_1,groundTruthTensor_1=batchShuffle(aggregate_1,groundTruth_1)
    pred_0,feat1_0,feat2_0,feat3_0=net(aggregateTensor_0)
    pred_1,feat1_1,feat2_1,feat3_1=net(aggregateTensor_1)
    loss=lossFunc(adapt,groundTruthTensor_0,pred_0,feat1_0,feat2_0,feat3_0,pred_1,feat1_1,feat2_1,feat3_1)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    del aggregateTensor_0,groundTruthTensor_0,aggregateTensor_1,groundTruthTensor_1,pred_0,feat1_0,feat2_0,feat3_0,pred_1,feat1_1,feat2_1,feat3_1,loss
    torch.cuda.empty_cache()
################################################################################
def evaluate(net,aggregate,groundTruth):
    with torch.no_grad():
        net=net.eval()
        aggregateTensor,groundTruthTensor=batchShuffle(aggregate,groundTruth)
        pred_0,feat1_0,feat2_0,feat3_0=net(aggregateTensor)
        loss=mse(pred_0,groundTruthTensor).item()
        del aggregateTensor,groundTruthTensor,pred_0,feat1_0,feat2_0,feat3_0
        torch.cuda.empty_cache()
        return loss
################################################################################
