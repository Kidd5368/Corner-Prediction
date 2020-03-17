import tensorflow as tf;
import pandas as pd;
import pprint
import numpy as np;
league='Premier League'
homemodel=tf.keras.models.load_model('/Users/kiddgu/PycharmProjects/CornerPrediction/AIhome Zscore.h5')
awaymodel=tf.keras.models.load_model('/Users/kiddgu/PycharmProjects/CornerPrediction/AIaway.h5')
totalmodel=tf.keras.models.load_model('/Users/kiddgu/PycharmProjects/CornerPrediction/AItotal Zscore.h5')
minusmodel=tf.keras.models.load_model('/Users/kiddgu/PycharmProjects/CornerPrediction/AIminus.h5')
totalgoalmodel=tf.keras.models.load_model('/Users/kiddgu/PycharmProjects/CornerPrediction/AItotalgoal Zscore.h5')

Teams=np.array([['Tottenham','Brighton']])
with open('/Users/kiddgu/PycharmProjects/CornerPrediction/Data/Fixture Data of '+league+' 2019.csv')as file:
    temp=pd.read_csv(file,header=None)
    file.close()
temp.dropna(axis=0,how='any',inplace=True)
with open('/Users/kiddgu/PycharmProjects/CornerPrediction/Data/Fixture Data of '+league+' 2019 clear.csv','w')as file:
    temp.to_csv(file,header=None,index=None)
    file.close()
with open('/Users/kiddgu/PycharmProjects/CornerPrediction/Data/Fixture Data of '+league+' 2019 clear.csv')as file:
    origin=pd.read_csv(file,header=None)
    file.close()
with open('/Users/kiddgu/PycharmProjects/CornerPrediction/Data/FinalData/'+league+' '+'2019Gu clear N.csv')as file:
    normalized=pd.read_csv(file,header=None)
    file.close()
with open('Data/temp.csv')as file:
    data=pd.read_csv(file,header=None)
def GetTeamData(home,away):
    for i in range(origin.shape[0]):
        if(origin.loc[i,1]==home):
            pprint.pprint(origin.loc[i,1])
            homeindex=i-1
            break
    for j in range(origin.shape[0]):
        if(origin.loc[j,42]==away):
            pprint.pprint(origin.loc[j,42])
            awayindex=j-1
            break
    homedata=normalized.loc[homeindex,0:33].copy().values
    awaydata=normalized.loc[awayindex,34:67].copy().values
    return np.concatenate((homedata,awaydata),axis=0)


def Evaluate():
    global Teams;
    temp=[]
    for i in range(Teams.shape[0]):
        d=GetTeamData(Teams[i][0],Teams[i][1])
        d=d.reshape(1,len(d))
        temp.append(d)
    for j in range(temp.__len__()):
        if(j==0):
            temparr=temp[j]
        else:
            temparr=np.concatenate((temparr,temp[j]),axis=0)
    pprint.pprint(homemodel(temparr))
    pprint.pprint(awaymodel(temparr))
    pprint.pprint(totalmodel(temparr))
    pprint.pprint(np.concatenate((Teams,totalmodel(temparr)),axis=1))

Evaluate()
'''
with open('Data/temp.csv')as file:
    a=pd.read_csv(file,header=None)
    file.close()
print(totalmodel(a.values))
with open('tempeva.csv','w')as file:
    b=pd.DataFrame(totalmodel(a.values).numpy())
    b.to_csv(file)
    file.close()
'''