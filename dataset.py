#il codice va eseguito su unv progettoNilmtk da cmd
#cd /home/andrea/miniconda3/bin/
#conda activate nilmtk-env
#python "/home/andrea/Desktop/digital adaptive/digital adaptive/codice/dataset.py"

#PER OGNI ABITAZIONE, ESTRARRE CONSUMO AGGREGATO E CONSUMO SINGOLO
#ELETTRODOMESTICI ANALIZZATI NEL PAPER WASHING MACHINE(WM), appliance(DW), appliance(MV), appliance(FG), appliance(KT)

from nilmtk import DataSet
import sys,struct,numpy,datetime,matplotlib,os

appliances=['washing machine','dish washer','microwave','fridge','kettle']
parameters=[[2500,20,1800,160],[2500,10,1800,1800],[3000,200,12,30],[300,50,60,12],[3100,2000,12,0]]
fullSet=DataSet('/home/andrea/Desktop/digital adaptive/digital adaptive/codice/data/ukdale.h5')
for build in fullSet.buildings:#.building lista di edifici
#for build in range(5,6):
    aggregate=fullSet.buildings[build].elec.mains()
    if(('power', 'active') in aggregate.available_columns()):
        aggregateData=next(aggregate.power_series(physical_quantity=('power', 'active'),sample_period=6,resample=True))
        for curApp in range(0,len(appliances)):
            try:#se non trova appliance
                appliance=fullSet.buildings[build].elec[appliances[curApp]]
                if(('power', 'active') in appliance.available_columns()):
                    #applianceData=next(appliance.power_series(physical_quantity=('power', 'active'),sample_period=6,resample=True))#pandas series
                    applianceAct=appliance.get_activations(on_power_threshold=parameters[curApp][1],
                                                             min_on_duration=parameters[curApp][2],
                                                             min_off_duration=parameters[curApp][3],
                                                             physical_quantity=('power', 'active'),
                                                             sample_period=6,
                                                             resample=True)
                    applianceData=next(appliance.power_series(physical_quantity=('power', 'active'),sample_period=6,resample=True))
                    #lista di series pandas, ogni series(tipo dizionario, .index da le chiavi(sempre in formato pandas)) contiene i punti che fanno parte di un'attivazione
                    x=[]#aggregato
                    y=[]#disaggregato
                    for curAct in range(0,len(applianceAct)):
                        startIndex=applianceAct[curAct].index[0]#primo istante dell'attivazione
                        dimAct=len(applianceAct[curAct].index)
                        dimTot=2*dimAct#bilancio al 50%
                        for curInd in range(0,dimTot):
                            offset=datetime.timedelta(seconds=curInd*6)
                            if(curAct==len(applianceAct)-1 or startIndex+offset<applianceAct[curAct+1].index[0]):
                                #se il nuovo punto che valuto non si sovrappone all'attivazione successiva(a causa del bilanciamento)
                                #l'or skippa se ultima attivazione
                                try:#se il dato ad un timestamp c'è non solleva eccezione
                                    xTmp=aggregateData[startIndex+offset]#primo controllo di esistenza dell'indice(sia su aggregato che su applicazione)
                                    yTmp=applianceData[startIndex+offset]
                                    if(numpy.isnan(xTmp)==False and numpy.isnan(yTmp)==False):#lo salvo solo se valido
                                        x.append(xTmp)
                                        y.append(yTmp)
                                except:None#print(sys.exc_info())
                            else:break
                        print(' '*os.get_terminal_size()[0]+'\rprocessed activation',curAct,'/',len(applianceAct),'of',appliances[curApp],'in building',build,'\r',end='')
                        print(end='')
                else:
                    print(' '*os.get_terminal_size()[0]+'\rnot found active power for '+appliances[curApp]+' in building '+str(build)+'\r',end='')
                numpy.save('/home/andrea/Desktop/digital adaptive/digital adaptive/codice/data/dataset/'+str(build)+'_'+appliances[curApp]+'_x',x)
                numpy.save('/home/andrea/Desktop/digital adaptive/digital adaptive/codice/data/dataset/'+str(build)+'_'+appliances[curApp]+'_y',y)
            except:
                #print(sys.exc_info())
                print(' '*os.get_terminal_size()[0]+'\rnot found',appliances[curApp],'in building',build,'\r',end='')
    else:
        print(' '*os.get_terminal_size()[0]+'\rnot found aggregate active power for build '+str(build)+'\r',end='')
               

    
    
    
        


        
    
    
