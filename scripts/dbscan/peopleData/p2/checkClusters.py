import pandas as pd

data = pd.read_csv('allData.out',sep=",")

newStrings = []
for index,row in data.iterrows():
	nS = str(row[1]) +','+str(row[2])
	newStrings += [nS]


keep = [] 
bad = []
for i in range(len(newStrings)):
	if newStrings[i] not in keep:
		keep += [newStrings[i]]
	else:
		bad += [newStrings[i]]
data2 = pd.read_csv('/home/CAM/dmcconnell/Opioid-LifeRhythm/dbscan/phase2_new.out',sep=",")
# ,lat,lon
keep2 = []
for idx,r in data2.iterrows():
	nSS = str(r.lat) +','+str(r.lon)
	keep2 += [nSS]

missed = []
for item in keep:
	if item not in keep2:
		missed += [item]
	
keepSmall = True if len(keep) > len(keep2) else False

if keepSmall:
	print("Individual files gave more unique clusters")
else:
	print("Single file gave more unique clusters")

print("Clusters in individual files: %d"%len(newStrings))
print("Duplicated clusters in individual files: %d"%len(bad))
print("Number of unique clusters in the indiviual files: %d"%len(keep))

print("Clusters in large file: %d"%len(keep2))
print("Number of different unique clusters in individual files: %d"%len(missed))

print("Number of overlapping clusters: %d"%(len(keep2) - len(missed)))
for item in missed:
	print(item)
