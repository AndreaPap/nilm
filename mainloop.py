import sys,os,numpy,copy
sys.path.append('.'+os.path.sep+'')
from utility import*
from rete import*
from torchsummary import summary
import torch.optim

def trainNet(elec,net,parPath,adapt,step,failLimit):
    learningRate=1e-3
    print(elec,learningRate)
    print('fixed mode')
    #summary(net,(1,600))
    print('working on:',device)
    optimizer=torch.optim.Adam(net.parameters(),lr=learningRate)
    aggregateTraining,groundTruthTraining,aggregateValidation,groundTruthValidation,aggregateTesting,groundTruthTesting=createSet(
        ['./data/dataset/5_'+elec+'_x.npy','./data/dataset/1_'+elec+'_x.npy'],
        ['./data/dataset/5_'+elec+'_y.npy','./data/dataset/1_'+elec+'_y.npy'],
        ['./data/dataset/2_'+elec+'_x.npy'],
        ['./data/dataset/2_'+elec+'_y.npy'],
        val=0)
    #standardizzazione dei dati
    #aggregateMean=522
    #aggregateStd=814
    #groundTruthMean=522
    #groundTruthStd=814
    aggregateMean=aggregateTraining.mean()
    aggregateStd=aggregateTraining.std()
    groundTruthMean=groundTruthTraining.mean()
    groundTruthStd=groundTruthTraining.std()
    
    aggregateTraining-=aggregateMean
    aggregateTraining/=aggregateStd
    aggregateValidation-=aggregateMean
    aggregateValidation/=aggregateStd
    aggregateTesting-=aggregateMean
    aggregateTesting/=aggregateStd
    groundTruthTraining-=groundTruthMean
    groundTruthTraining/=groundTruthStd
    groundTruthValidation-=groundTruthMean
    groundTruthValidation/=groundTruthStd
    groundTruthTesting-=groundTruthMean
    groundTruthTesting/=groundTruthStd
    #
    iterations=1500
    #step=10
    #failLimit=5
    #if(adapt==False):failLimit=5
    #else:failLimit=10
    lossTrack=[]
    fail=0
    try:
        for cur in range(0,iterations+1):
            train(net,optimizer,aggregateTraining,groundTruthTraining,aggregateTesting,groundTruthTesting,adapt)
            if(cur%step==0):
                print('#'*10+'\tITERATION   '+str(cur)+'/'+str(iterations)+'\t'+'#'*10)
                lossTr=evaluate(net,aggregateTraining,groundTruthTraining)
                #lossVa=evaluate(net,xValidation,yValidation)
                lossTe=evaluate(net,aggregateTesting,groundTruthTesting)
                print('training loss:',round(lossTr,3))
                #print('validation loss:',round(lossVa,3))
                print('testing loss:',round(lossTe,3))
                if(len(lossTrack)==0 or lossTe<min(lossTrack)):
                    print('best testing loss, saving network')
                    torch.save(net.state_dict(),parPath)
                    fail=0
                else:fail+=1
                if(fail==failLimit):raise Exception('Net stop to learn')
                lossTrack.append(lossTe)
                #del lossTr,lossVa,lossTe
                del lossTr,lossTe
                torch.cuda.empty_cache()
                print('#'*50)
    except:print(sys.exc_info())#uscita forzata con eccezione
    print('best testing loss:',round(min(lossTrack),3),'@',lossTrack.index(min(lossTrack))*step)
    print('#'*10+'\tEND\t'+'#'*10)
    
def compareNet(elec,listNet,listTitle,dataPath):
    with torch.no_grad():
        aggregateTraining,groundTruthTraining,aggregateValidation,groundTruthValidation,aggregateTesting,groundTruthTesting=createSet(
            ['./data/dataset/5_'+elec+'_x.npy','./data/dataset/1_'+elec+'_x.npy'],
            ['./data/dataset/5_'+elec+'_y.npy','./data/dataset/1_'+elec+'_y.npy'],
            ['./data/dataset/2_'+elec+'_x.npy'],
            ['./data/dataset/2_'+elec+'_y.npy'],
            val=0)
        #standardizzazione dei dati
        aggregateMean=aggregateTraining.mean()
        aggregateStd=aggregateTraining.std()
        groundTruthMean=groundTruthTraining.mean()
        groundTruthStd=groundTruthTraining.std()
        #aggregateMean=522
        #aggregateStd=814
        #groundTruthMean=522
        #groundTruthStd=814
    
        aggregateTraining-=aggregateMean
        aggregateTraining/=aggregateStd
        aggregateValidation-=aggregateMean
        aggregateValidation/=aggregateStd
        aggregateTesting-=aggregateMean
        aggregateTesting/=aggregateStd
        groundTruthTraining-=groundTruthMean
        groundTruthTraining/=groundTruthStd
        groundTruthValidation-=groundTruthMean
        groundTruthValidation/=groundTruthStd
        groundTruthTesting-=groundTruthMean
        groundTruthTesting/=groundTruthStd
        #
        aggregate,groundTruth=batchShuffle(aggregateTesting,groundTruthTesting)
        listPred=[]
        for net in listNet:
            pred,feat1,feat2,feat3=net(aggregate)
            pred*=groundTruthStd
            pred+=groundTruthMean
            listPred.append(copy.deepcopy(pred))
        aggregate*=aggregateStd
        aggregate+=aggregateMean
        groundTruth*=groundTruthStd
        groundTruth+=groundTruthMean
        overlapPlot([aggregate,groundTruth]+listPred,['Aggregate','Ground truth']+listTitle,dataPath,elec+' global')
        for cur in range(0,10):
            aggregate,groundTruth=batchShuffle(aggregateTesting,groundTruthTesting,batchSize=1)
            listPred=[]
            for net in listNet:
                pred,feat1,feat2,feat3=net(aggregate)
                pred*=groundTruthStd
                pred+=groundTruthMean
                listPred.append(copy.deepcopy(pred))
            aggregate*=aggregateStd
            aggregate+=aggregateMean
            groundTruth*=groundTruthStd
            groundTruth+=groundTruthMean
            overlapPlot([aggregate,groundTruth]+listPred,['Aggregate','Ground truth']+listTitle,dataPath,elec+' local '+str(cur))
        print('saved plot')
        file=open(dataPath+'report.txt','w')
        file.write('report '+elec+'\n\n')
        for cur in range(0,len(listNet)):
            aggregate,groundTruth=batchShuffle(aggregateTesting,groundTruthTesting,windowSize=600,batchSize=100)
            pred,feat1,feat2,feat3=listNet[cur](aggregate)
            groundTruthDenorm=groundTruth*groundTruthStd
            groundTruthDenorm+=groundTruthMean
            #mse=(pred-groundTruth).pow(2).mean()
            pred*=groundTruthStd
            pred+=groundTruthMean
            mae=(pred-groundTruthDenorm).abs().mean([1,2])#lavoro sulle ultime due dimensioni, il batch lo uso per media e deviazione standard
            aggregate,groundTruth=batchShuffle(aggregateTesting,groundTruthTesting,windowSize=3600,batchSize=100)
            pred,feat1,feat2,feat3=listNet[cur](aggregate)
            groundTruthDenorm=groundTruth*groundTruthStd
            groundTruthDenorm+=groundTruthMean
            pred*=groundTruthStd
            pred+=groundTruthMean
            sae=(pred.mean([1,2])-groundTruthDenorm.mean([1,2])).abs()
            file.write(listTitle[cur]+':\n\tmae: '+str(round(mae.mean().item(),1))+u'\u00b1'+str(round(mae.std().item(),1))+\
            '\n\tsae: '+str(round(sae.mean().item(),1))+u'\u00b1'+str(round(sae.std().item(),1))+'\n')
        file.close()
        print('saved metrics')
    
    
