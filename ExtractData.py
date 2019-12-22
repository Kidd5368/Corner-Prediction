import pandas as pd;
import pprint;
import random;
import numpy as np;
league='Serie A'
year='2019'
with open('Data/Fixture Data of '+league+' '+year+'.csv')as file:
    data=pd.read_csv(file,header=None)
    file.close()

if(data.shape[1]==89):
    data1=data.drop(axis=1,columns=[0,1,2,3,42,44,43,83,84,85,86,87,88],inplace=False)#去除多余的队名，队id，角球,进球等等
elif(data.shape[1]==87):
    data1=data.drop(axis=1,columns=[0,1,2,3,42,44,43,83,84,85,86],inplace=False)#去除多余的队名，队id，角球,进球等等
else:
    print('error')

data1.drop(axis=0,index=[0],inplace=True)
pprint.pprint(data1)
with open('Data/'+league+' '+year+'Gu.csv','w')as file:
    data1.to_csv(file,header=None,index=None)
    file.close()
with open('Data/'+league+' '+year+'Gu.csv')as file:
    data2=pd.read_csv(file,header=None)
    pprint.pprint(data2.loc[:,[28,29,66,67]])
data2.loc[:,28]=data2.loc[:,26]/data2.loc[:,24]#19赛季控球率缺失
data2.loc[:,29]=data2.loc[:,27]/data2.loc[:,25]
data2.loc[:,66]=data2.loc[:,64]/data2.loc[:,62]
data2.loc[:,67]=data2.loc[:,65]/data2.loc[:,63]
data2.drop(axis=1,columns=[4,5,31,32,42,43,69,70],inplace=True)#去除与其他特征有线性关系的特征和无关特征
with open('Data/'+league+' '+year+'Gu clear.csv','w')as file:
    data2.to_csv(file,header=None,index=None)
    file.close()
with open('Data/'+league+' '+year+'Gu clear.csv')as file:
    data2=pd.read_csv(file,header=None)
    pprint.pprint(data2.loc[:,[26,27]])

data3=(data2-data2.mean())/(data2.max()-data2.min())#归一化
with open('Data/FinalData/'+league+' '+year+'Gu clear N.csv','w')as file:
    data3.to_csv(file,header=None,index=None)
    file.close()
with open('Data/FinalData/'+league+' '+year+'Gu clear N.csv')as file:
    data3=pd.read_csv(file,header=None)
    file.close()
if(data.shape[1]==89):
    temp=np.concatenate((data3.copy().values,data.loc[1:,83:86].copy().values.astype(np.float)),axis=1)#沾上角球和进球
elif(data.shape[1]==87):
    temp = np.concatenate((data3.copy().values, data.loc[1:,83:86].copy().values.astype(np.float)), axis=1)
else:
    print('error')
temp=pd.DataFrame(temp)
with open('Data/FinalData/'+league+' '+year+'Gu clear N.csv','w')as file:
    temp.to_csv(file,header=None,index=None)
    file.close()
with open('Data/FinalData/'+league+' '+year+'Gu clear N.csv')as file:
    data4=pd.read_csv(file,header=None)
    file.close()
pprint.pprint(data4)#之后还要手动把角球数和进球数粘上去


'''
pprint.pprint(data.loc[:,[32,70]])
data.drop(axis=1,columns=[4,5,31,32,42,43,69,70],inplace=True)
pprint.pprint(data)
with open('Data/'+year+'clear.csv','w')as file:
    data.to_csv(file,header=None,index=None)
    file.close()
with open('Data/'+year+'clear.csv')as file:
    data1=pd.read_csv(file,header=None)
    pprint.pprint(data1)

'''

