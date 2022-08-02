import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import sys
from sklearn import metrics
from multiprocessing import Pool

def clusterData(listInfo):
    inFile = listInfo[0]
    dist = listInfo[1]
    minSamp = listInfo[2]

    df = pd.read_csv(inFile,skiprows=[0],names=["id",'time',"lat","long","flag"])#('phase2_new.csv',encoding = 'unicode_escape')
    df = df[df.flag == 0]
    X = np.array(df[['lat','long']])

    clustering = DBSCAN(eps=dist/6371.0, min_samples=minSamp, algorithm='ball_tree',metric='haversine',n_jobs=-1).fit(np.radians(X)) # replace the 0.005 to 0.025 for distance of 25 m


    # dataframe = pd.DataFrame()
    #
    # dataframe["lat"] = df['lat']
    # dataframe["lon"] = df['long']
    # dataframe["label"] = clustering.labels_
    # dataframe["visited"] = False

    silo = metrics.silhouette_score(X, clustering.labels_, metric='euclidean')
    
    print("Parameters --> eps: {} samples: {} \n\t Score: {}".format(dist,minSamp,silo))
    return silo


    # my_lat = []
    # my_lon = []
    # i = 0
    # count = 0
    # lat = 0
    # lon = 0
    #
    # for label in clustering.labels_:
    # #for i in range((dataframe.lable.max())+1):
    #     ll = dataframe[dataframe["label"] == label]
    #     if ll.visited.iloc[0] == False:
    #         for i,r in dataframe.iterrows():
    #        #for kmp in range(dataframe.shape[0]):
    #             if r.label == label:
    #                  lat = lat + r.lat
    #                  lon = lon + r.lon
    #                  count = count +1
    #
    #         indices = dataframe[dataframe["label"] == label].index
    #
    #         dataframe.visited.loc[indices] = True
    #
    #         my_lat.append(lat/count)
    #         my_lon.append(lon/count)
    #
    #         count = 0
    #         lat = 0
    #         lon = 0
    #
    # df_id_lat_lon = pd.DataFrame()
    # df_id_lat_lon ["lat"] = my_lat
    # df_id_lat_lon ["lon"] = my_lon
    #
    # tmp = sys.argv[1]
    # tmp = tmp.split('/')
    # splitF = tmp[-1].split('.')
    # outName = splitF[0] + '_EPS_%s_Samp_%s.out'%(dist,minSamp)
    # print(outName)
    #
    # df_id_lat_lon.to_csv(outPath+ '/' + outName)#("out2.csv")

if __name__ == '__main__':
    inputFile = sys.argv[1]
    #outputPath = sys.argv[2]
    print("WORING ON FILE: {}".format(inputFile))
    search = []

    for i in range(2, 57, 2):
        dist = i / 1000
        for j in range(3, 162, 2):
            search += [[inputFile,dist,j]]

    listOfScores = []
    for item in search:
        listOfScores += [clusterData(item)]

    bestScore = np.argmax(listOfScores)
    print("\n\nTop Parameters")
    searched = search[bestScore]
    dist = searched[1]
    minSamp = searched[2]
    silo = listOfScores[bestScore]
    print("Parameters --> eps: {} samples: {} \n\t Score: {}".format(dist, minSamp, silo))
