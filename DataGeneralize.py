import pandas as pd;
import random;
import numpy as np;
import pprint;
league='Serie A'
year='2019'
fois=1
variance=0.05

with open('Data/FinalData/'+league+' '+year+'Gu clear N.csv')as file:
    data=pd.read_csv(file,header=None)
    a=data.shape[0]
    data.dropna(axis=0,how='any',inplace=True)#去除2019年还没打的比赛
    print(a-data.shape[0])
    file.close()
with open('Data/FinalData/'+league+' '+year+'Gu clear N.csv','w')as file:
    data.to_csv(file,header=None,index=None)
    file.close()
with open('Data/FinalData/'+league+' '+year+'Gu clear N.csv')as file:
    data=pd.read_csv(file,header=None)
    file.close()
finaldata=data.copy(deep=True)
print(finaldata.shape)
for k in range(fois):
    tempdata=data.copy(deep=True)
    print(tempdata.shape)
    for i in range(tempdata.shape[0]):
        for j in range(tempdata.shape[1]):
            tempdata.loc[i,j]+=random.gauss(0,variance)
    finaldata=finaldata.append(tempdata,ignore_index=True)
with open('Data/FinalData/'+league+' '+year+'Gu clear NG.csv','w')as file:
    finaldata.to_csv(file,header=0,index=None)
    file.close()
pprint.pprint(finaldata)