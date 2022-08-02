import pandas as pd
import numpy as np 
from sklearn.cluster import DBSCAN
import sys
df = pd.read_csv(sys.argv[1],skiprows=[0],names=["id",'time',"lat","long","flag"])#('phase2_new.csv',encoding = 'unicode_escape')
df = df[df.flag == 0]
X = np.array(df[['lat','long']])
clustering = DBSCAN(eps=0.014/6371.0, min_samples=3, algorithm='ball_tree',metric='haversine',n_jobs=20).fit(np.radians(X)) # replace the 0.005 to 0.025 for distance of 25 m


dataframe = pd.DataFrame()

dataframe["lat"] = df['lat']
dataframe["lon"] = df['long']
dataframe["label"] = clustering.labels_
dataframe["visited"] = False

my_lat = []
my_lon = []
i = 0
count = 0
lat = 0
lon = 0

for label in clustering.labels_:
#for i in range((dataframe.lable.max())+1):
    ll = dataframe[dataframe["label"] == label]
    if ll.visited.iloc[0] == False:
        for i,r in dataframe.iterrows():
       #for kmp in range(dataframe.shape[0]):
            if r.label == label:
                 lat = lat + r.lat
                 lon = lon + r.lon
                 count = count +1

        indices = dataframe[dataframe["label"] == label].index

        dataframe.visited.loc[indices] = True

        my_lat.append(lat/count)
        my_lon.append(lon/count)

        count = 0
        lat = 0
        lon = 0

df_id_lat_lon = pd.DataFrame()
df_id_lat_lon ["lat"] = my_lat
df_id_lat_lon ["lon"] = my_lon

tmp = sys.argv[1]
tmp = tmp.split('/')
splitF = tmp[-1].split('.')
outName = splitF[0] + '.out'
print(outName)

df_id_lat_lon.to_csv(sys.argv[2] + '/' + outName)#("out2.csv")
