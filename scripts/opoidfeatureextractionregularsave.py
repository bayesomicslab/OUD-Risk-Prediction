"""
@author: kaustubhprabhu
"""
import pandas as pd
import numpy as np
from os import listdir
from multiprocessing import Pool
from subprocess import call
import sys
import math
from math import log
from decimal import Decimal
import copy
from collections import Counter

sys.path.insert(1, '/Users/kaustubhprabhu/Downloads/Opioid-LifeRhythm/scripts')
from KDtree import *

tthreshold=60 # threshold: 60 = 60 sec, one sample can only represent 1 mins stay
z=0
epochtime = 1420372800 # 2015/01/03  12:00 am, used to calculate hour of the day quickly
eps = 0.0001#:0.0001:0.001
minpts = 5 #2:1:30
#Read data - for all users one by one
re=[]
#for zz = 1:size(gpsData,2) # Last column has users id



#features stored here

# Creating an empty Dataframe with column names only
gpsFeatures = pd.DataFrame(columns=['uid', 'stime','var', 'avgspd', 'ent', 'lgent', 'home', 'transtime', 'totdist', 'routInd', 'indgr', 'outdgr', 'uniqueC','uniTC', 'tsc1',  'tsc2', 'tsc3', 'tsc4', 'tsc5', 'tsc6', 'tsc7',  'tsc8', 'tsc9' ])
ff = open('/Users/kaustubhprabhu/Downloads/Opioid-LifeRhythm/scripts/locationsTree.pickle', 'rb')
newTree = pickle.load(ff)
import glob
path = '/Users/kaustubhprabhu/Downloads/p1/*.csv'
for fname in glob.glob(path):
        routineStore = []
        for i in range (0,7):
            column_names = ["locNm", "q", "timespent"]
            routineStore.insert(i, pd.DataFrame(columns = column_names))
        print(fname)
        tmpdata = pd.read_csv(fname)

        tada = np.floor((tmpdata.iloc[:,1]-epochtime)/86400)
        tada = tada.astype('int32')
        tmpdata.insert(5,'kmp', tada)
        wd = np.mod(tmpdata.iloc[:,5],7)+1
        wd = wd.astype('int32')
        tmpdata.insert(6,'wekday', wd)
        days = tmpdata.iloc[:,5].unique()

        th=1 # %12km/hour
        lenth = len(tmpdata)
        tmpdata["motion"] = 0
        for l in range(2,lenth):
            fP1Lat = tmpdata.iloc[l-1,2]
            fP1Lon = tmpdata.iloc[l-1,3]
            fP2Lat = tmpdata.iloc[l,2]
            fP2Lon = tmpdata.iloc[l,3]
            fRadLon1 = (np.pi/180)*(fP1Lon)
            fRadLon2 = (np.pi/180)*(fP2Lon)
            fRadLat1 = (np.pi/180)*(fP1Lat)
            fRadLat2 = (np.pi/180)*(fP2Lat)
            fD1 = abs(fRadLat1 - fRadLat2)
            fD2 = abs(fRadLon1 - fRadLon2)
            fP = (math.sin(fD1/2)**2) + (math.cos(fRadLat1) * math.cos(fRadLat2) * (math.sin(fD2/2)**2))
            d = 6371.137 * 2 * math.asin(math.sqrt(fP))
            time = (tmpdata.iloc[l,1]-tmpdata.iloc[l-1,1])
            if ((d*3600)/time > th):
                tmpdata.loc[l,'motion']=1
                tmpdata.loc[l-1,'motion']=1

        for day in range(0,len(days)):
                z=z+1
                #######################
                userData = (tmpdata[tmpdata['kmp'] == days[day]].reset_index(drop=True)).copy()
                ########################

                if (len(userData) < 2): #% if no data found, ignore move one
                    print("hi no data for the day")
                    continue

                #in case the time is in microsecond
                periodts = userData.iloc[0,1].copy()

                uid = userData.iloc[0,0]#; % uid
                stime = userData.iloc[0,1]#; % start time
                userData1 = userData.iloc[:, [1,2,3,6,23]].copy() #% [timestamp, lat, long, weekday, moving state]
                #%% Time spent calculation (preprocessing step)
                #% Calculate how much time user spent on a perticular location
                #% Traverse through all location samples
                dummyarray = np.empty((userData1.shape[0],1))
                dummyarray[:] = np.nan
                timeSpent =pd.DataFrame(dummyarray) #% clear timeSpent
                timeSpent.iloc[0,0] = tthreshold
                for i in range( 1,userData1.shape[0]):
                    latDiff =abs(userData1.iloc[i-1,1] - userData.iloc[i,1])
                    longDiff = abs(userData1.iloc[i-1,2] - userData1.iloc[i,2])
                    timeSpent.iloc[i,0] = userData1.iloc[i,0] - userData1.iloc[i-1,0]
                    #print(timeSpent.iloc[i,0])
                    if timeSpent.iloc[i,0] < 0 :
                        print('something is fishy, see code line 91')
                        #exit(5)# % pause, in case timespent seems incorrect

                    if timeSpent.iloc[i,0] > tthreshold: #% missing data case
                        timeSpent.iloc[i,0] = tthreshold# % android: 10 minutes  ios: 1 minute

                #% end for - this will give us time spent b/w consecutive long/lat
                userData1.insert(5,'timespent',timeSpent.iloc[:,0].copy())#Collumn 6

                #% add as a column % [timestamp, lat, long, weekday, mins_between_traces]
                # %% Feature 1 = Variance
                # % First feature - Location variance - that measures the variability in the
                # % Calculate statistical variance of longitude and latitude
                varLat = np.var(userData1.iloc[:,1])
                varLong = np.var(userData1.iloc[:,2])
                addlog = Decimal(varLat + varLong)
                k = addlog.ln()#; % Nautral Log
                locVar = float(k)
                #print(locVar)
                feature1 = locVar
                 #%% Feature 2 = Entropy
                #% Entropy - to measure the variability of the time the subject spent at a
                #

                userData1['locNm']= "outlier"
                userData1['locTy']= "outlier"
                #print

                for i in range(1,userData1.shape[0]):


                    sam2 = newTree.queryTree([userData1.iloc[i,1], userData1.iloc[i,2]])
                    #print(userData1.iloc[i,1], userData1.iloc[i,2])
                    sam1 = newTree.queryTree([userData1.iloc[i-1,1], userData1.iloc[i-1,2]])
                    #print (userData1.iloc[i-1,1], userData1.iloc[i-1,2])
                    if(sam1.Name()==sam2.Name()):
                        #print(sam1.Type())
                        userData1.loc[i-1,'locNm'] = copy.deepcopy(sam1.Name())
                        userData1.loc[i-1, 'locTy'] =copy.deepcopy( sam1.Type())
                        userData1.loc[i,'locNm'] = copy.deepcopy(sam2.Name())
                        userData1.loc[i, 'locTy']= copy.deepcopy(sam2.Type())



                #clusnType.fillna("outlier")

                #print(userData1['locNm'])


                #% Calculate Entropy
                ent = 0
                t =  np.mod(((userData1.iloc[:,0])-epochtime)/3600,24)
                col = userData1.shape[1]
                userData1.insert(6,'q',t.loc[:].copy())
                #print(userData1['q'])





                # %% Feature 4 = Number of Unique Clusters
                #% Number of unique clusters
                uniClus = np.unique(userData1['locNm'])#; % Cluster number % ASMA CHANGED
                numUniClust = len(uniClus)#; % Number of unique clusters
                uniClusTy = np.unique(userData1['locTy'])#; % Cluster type number % ASMA CHANGED
                numUniClustTy = len(uniClusTy)#; % Number of unique cluster types
                #gpsFeatures(z,10) = I; #% outlier already removed from cluster

                totTimeClusdat = userData1[ userData1['locNm'] !=  "outlier"].copy()#; #% total time spend @ "all" meaningful clusters
                #print(totTimeClusdat['q'])
                totTimeClus = sum(totTimeClusdat['timespent'][:])
                if numUniClustTy > 1:

                    tsumTimeTy = pd.DataFrame(np.zeros((numUniClustTy,1)))
                    tsc = pd.DataFrame(np.zeros((9,1)))

                    ty = ["Outdoors & Recreation","Professional & Other Places" , "Shop & Service", "Food", "Travel & Transport", "Residence", "College & University", "Arts & Entertainment", "Nightlife Spot"]
                    tsc['typ'] = ty

                    for i in range(0,numUniClustTy): #% total time spend in the cluster

                        if (uniClusTy[i] != "outlier"):
                            tmesptindTy =  userData1[userData1['locTy'] == uniClusTy[i]].copy()#; % find the cluster index
                            #print(tmesptindTy['timespent'])
                            tsumTimeTy.iloc[i] = sum(tmesptindTy['timespent'][:]) #; % total time spent in a perticular cluster i.e., ith cluster

                            tsc.loc[tsc["typ"]== uniClusTy[i],0] = float(tsumTimeTy.iloc[i]/totTimeClus)#; % Percentage of time spent @ cluster i



                if numUniClust > 1: #% if atleast 2 clusters found
                    maxa = 0
                    tsumTime = pd.DataFrame(np.empty((numUniClust,1)))
                    sumTime = pd.DataFrame(np.empty((numUniClust,1)))
                    f = 1#; % Variable that finally will have home cluster information
                    #nightcluster=[]#;


                    for i in range(0,numUniClust): #% total time spend in the cluster

                        if (uniClus[i] != "outlier"):
                            tmesptind =  userData1[userData1['locNm'] == uniClus[i]].copy()#; % find the cluster index

                            tsumTime.iloc[i] = sum(tmesptind['timespent'][:]) #; % total time spent in a perticular cluster i.e., ith cluster

                            sumTime.iloc[i] = tsumTime.iloc[i]/totTimeClus#; % Percentage of time spent @ cluster i

                            ent = ent + ( float(sumTime.iloc[i]) * (np.log(float(sumTime.iloc[i]))))#; % calculate entropy

                        if (uniClus[i] != "outlier"):


                            idtmp = tmesptind[tmesptind['q'] >= 0].copy()#); % Time between 0 and 6
                            idset = idtmp[idtmp['q']<=6].copy()
                            if(( sum(tmesptind['timespent'])/sum(userData1.loc[:,'q'])) > 0.1 ):
                                hometime = sum(idset['timespent'])

                                if hometime > maxa:
                                    maxa = hometime
                                    f = i#; % By the end of iterations, f will have the cluster marked as home
                        else:
                            hmeFeature = tmesptind['timespent']/userData1.shape[0]
                    ent = - ent
                    #%% Feature 3 = Normalized Entropy
                    normEnt = ent/(np.log(numUniClust))
                    #%% Feature 5 = Home stay - cluster which have the time between 12am to 6am
                    homeTime = (sumTime.iloc[f])#;%/sum(sumTime);
                else:
                    ent = 0
                    normEnt = 0
                    homeTime = 1
                #%% Feature 6 = Transition time
                #% Transition time = number of samples in transition/ total number of
                #% samples
                k=userData1[userData1.iloc[:,4]== 1].copy()
                movingSamples = sum(k['timespent'])
                AllSamples = sum(userData1['timespent'])
                transition = movingSamples/ AllSamples
                #%% Feature 7 = Total Distance
                #% Total distance traveled in km
                sumdistkm = 0
                for j in range (1, (userData1.shape[0])):

                    lat1=userData1.iloc[j-1,1]*np.pi/180
                    lat2=userData1.iloc[j,1]*np.pi/180
                    lon1=userData1.iloc[j-1,2]*np.pi/180
                    lon2=userData1.iloc[j,2]*np.pi/180
                    deltaLat=lat2-lat1
                    deltaLon=lon2-lon1
                    a=np.sin((deltaLat)/2)**2 + np.cos(lat1)*np.cos(lat2) * np.sin(deltaLon/2)**2
                    c=2*math.atan2(np.sqrt(a),np.sqrt(1-a));
                    d1km=6371*c #   %Haversine distance


                    sumdistkm = sumdistkm + d1km



                #%% Feaure 8 = Average Moving Speed
                avgSpeed = sumdistkm/((userData1.iloc[((userData1.shape[0])-1),0]-userData1.iloc[0,0]))#%average moving speed  unit:km/s

                # Feature: Hubbiness
                #hubbiness will be calculated
                inDegree = pd.DataFrame(np.zeros((numUniClust,1)))
                outDegree = pd.DataFrame(np.zeros((numUniClust,1)))
                avgoutDegree = 0
                avginDegree = 0
                for i in range(0,numUniClust):
                    inlist = []
                    outlist = []
                    flg = 0
                    flg1 = 0
                    count =0
                    count1 = 0

                    if (uniClus[i] != "outlier"):
                        prevj = 0
                        for j in (userData1[userData1['locNm'] == uniClus[i]].index.tolist()):


                        #print("hi")
                                if (j>0 and userData1.loc[j,'locNm'] != userData1.loc[j-1,'locNm']):
                                    #if(flg1 == 1 or userData1.loc[j-1, 'locNm'] != "outlier"):
                                    for kmp in range(j-1, prevj, -1):
                                            if(userData1.loc[kmp, 'locNm'] != "outlier"):

                                                    if(not(userData1.loc[kmp,'locNm'] in inlist)):
                                                        inlist.append(userData1.loc[kmp, 'locNm'])
                                                    else:
                                                        flg =1
                                                    break
                                    if flg == 1:
                                                 count = count+1


                                if  (userData1.loc[prevj,'locNm'] != userData1.loc[prevj+1,'locNm']):
                                    for kmp in range(prevj,j):
                                            if(userData1.loc[kmp, 'locNm'] != "outlier"):

                                                    if(not(userData1.loc[kmp,'locNm'] in outlist)):
                                                        outlist.append(userData1.loc[kmp, 'locNm'])
                                                    else:
                                                        flg1 =1
                                                    break
                                    if flg1 == 1:
                                                 count1 = count1 +1
                                prevj = j
                        if (prevj < userData1.shape[0]-1) and (userData1.loc[prevj,'locNm'] != userData1.loc[prevj+1,'locNm']):
                                    for kmp in range(prevj,userData1.shape[0]):
                                            if(userData1.loc[kmp, 'locNm'] != "outlier"):

                                                    if(not(userData1.loc[kmp,'locNm'] in outlist)):
                                                        outlist.append(userData1.loc[kmp, 'locNm'])
                                                    else:
                                                        flg1 =1
                                                    break
                                    if flg1 == 1:
                                                 count1 = count1 +1
                        inDegree.iloc[i] = len(inlist)
                        outDegree.iloc[i] = len(outlist)
                if numUniClust > 2:
                    avginDegree = ((int(sum(inDegree.iloc[:,0]))+1)/(numUniClust-1))
                    avgoutDegree =((int(sum(outDegree.iloc[:,0]))+1)/(numUniClust-1))
                #print(avginDegree, avgoutDegree)

                userNout = (userData1[userData1['locNm'] != "outlier"].reset_index(drop=True)).copy()

                #Routine Index
                if(routineStore[userData1.iloc[0,3]-1].empty):
                    print("hi")
                    ri= 1
                else:
                  print("kmp123")
                  ri = 0
                  ct = 0
                  #routineStore[userData1.iloc[0,3]-1] = userData1.iloc[:, [5,6,7]].copy()
                  k = routineStore[userData1.iloc[0,3]-1]



                  t = k.loc[k['q'] > 21,'locNm'].value_counts().index.tolist()
                  p = userNout.loc[userNout['q'] > 21,'locNm'].value_counts().index.tolist()
                  if(len(t)>0 and len(p)>0):
                      ri = ri + len(set(t) & set(p))/max(len(t),len(p))
                      ct = ct +1

                  t = k.loc[k['q'] <6,'locNm'].value_counts().index.tolist()
                  p = userNout.loc[userNout['q'] < 6,'locNm'].value_counts().index.tolist()
                  if(len(t)>0 and len(p)>0):
                      ri = ri + len(set(t) & set(p))/max(len(t),len(p))
                      ct = ct +1


                  t = k.loc[(21>= k['q']) & k['q'] > 18,'locNm'].value_counts().index.tolist()
                  p = userNout.loc[(21>= userNout['q']) & userNout['q']> 18,'locNm'].value_counts().index.tolist()
                  if(len(t)>0 and len(p)>0):
                      ri = ri + len(set(t) & set(p))/max(len(t),len(p))
                      ct = ct +1

                  t = k.loc[(18>= k['q']) & k['q'] > 15,'locNm'].value_counts().index.tolist()
                  p = userNout.loc[(18>= userNout['q']) & userNout['q'] > 15,'locNm'].value_counts().index.tolist()
                  if(len(t)>0 and len(p)>0):
                      ri = ri + len(set(t) & set(p))/max(len(t),len(p))
                      ct = ct +1



                  t = k.loc[(15>= k['q']) & k['q'] > 12,'locNm'].value_counts().index.tolist()
                  p = userNout.loc[(15>= userNout['q']) & userNout['q']> 12,'locNm'].value_counts().index.tolist()
                  if(len(t)>0 and len(p)>0):
                      ri = ri + len(set(t) & set(p))/max(len(t),len(p))
                      ct = ct +1

                  t = k.loc[(12>=  k['q']) & k['q'] > 9,'locNm'].value_counts().index.tolist()
                  p = userNout.loc[(12>=userNout['q'] )& userNout['q'] > 9,'locNm'].value_counts().index.tolist()
                  if(len(t)>0 and len(p)>0):
                      ri = ri + len(set(t) & set(p))/max(len(t),len(p))
                      ct = ct +1

                  t = k.loc[(9>= k['q']) & k['q'] > 6,'locNm'].value_counts().index.tolist()
                  p = userNout.loc[(12>= userNout['q']) & userNout['q'] > 9,'locNm'].value_counts().index.tolist()
                  if(len(t)>0 and len(p)>0):
                      ri = ri + len(set(t) & set(p))/max(len(t),len(p))
                      ct = ct +1
                  if(ct != 0) :
                      ri = ri/ct

                routineStore[userData1.iloc[0,3]-1] = userNout.iloc[:, [5,6,7]].copy()
                tsc
                gpsFeatures = gpsFeatures.append(pd.Series([uid, stime, locVar, avgSpeed , ent, normEnt, float(homeTime), transition, sumdistkm, ri, avginDegree, avgoutDegree, numUniClust, numUniClustTy, tsc.iloc[0,0], tsc.iloc[1,0], tsc.iloc[2,0], tsc.iloc[3,0], tsc.iloc[4,0], tsc.iloc[5,0], tsc.iloc[6,0], tsc.iloc[7,0], tsc.iloc[8,0]], index=gpsFeatures.columns ), ignore_index=True) #sumTimeTy ])
                #[(columns=['uid', 'stime','var', 'avgspd', 'ent', 'lgent', 'home', 'transtime', 'totdist', 'Istart', 'routInd', 'indgr', 'outdgr', 'uniqueC','uniTC'])
                #print(gpsFeatures)
gpsFeatures.to_csv ('gpsp1finaltime.csv', index = False, header=True)

