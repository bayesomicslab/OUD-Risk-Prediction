import pandas as pd
import numpy as np 
from sklearn.cluster import OPTICS
import sys
df = pd.read_csv(sys.argv[1],encoding = 'unicode_escape')#('phase2_new.csv',encoding = 'unicode_escape')
X = np.array(df[['C','D']])
clustering = OPTICS(eps=0.005/6371.0, min_samples=10, algorithm='ball_tree',metric='haversine',n_jobs=-1).fit(np.radians(X)) # replace the 0.005 to 0.025 for distance of 25 m


dataframe = pd.DataFrame()

dataframe["lat"]= df['C']
dataframe["lon"]= df['D']
dataframe["lable"]= clustering.labels_


my_lat = []
my_lon = []
i = 0
kmp =0
count = 0
lat = 0
lon =0
for i in range((dataframe.lable.max())+1):
   for kmp in range(dataframe.shape[0]):
        if dataframe["lable"][kmp] == i:
             lat = lat + dataframe["lat"][kmp]
             lon = lon + dataframe["lon"][kmp]
             count = count +1
             kmp = kmp+1
   my_lat.append(lat/count)  
   my_lon.append(lon/count)
   kmp =0
   count =0
   i = i+1
   lat = 0
   lon = 0

df_id_lat_lon = pd.DataFrame()
#df_id_lat_lon ["id"] = list(range(0,(dataframe.lable.max()+1)))
df_id_lat_lon ["lat"] = my_lat
df_id_lat_lon ["lon"] = my_lon


tmp = sys.argv[1]
splitF = tmp.split('.')
outName = splitF[0] + '.out'
print(outName)

df_id_lat_lon.to_csv(outName)#("out2.csv")
