from os import listdir
import pandas as pd

def getlines(f):
	ff = open(directory+f,'r')
	cc = 0
	for line in ff:
		cc += 1

	return cc

if __name__ == '__main__':

	directory = 'out2/'
	files = listdir(directory)
	
	files = [item for item in files if ".out" in item]

	looked = []	
	for item in files:
		counter = 0
		numFs = 0
		head = item.split('_')
		
		if head[0] not in looked:
			for obj in files:
				if head[0] in obj:
					counter += getlines(obj)	
					numFs += 1
			print("Average Change in clusters {} for the files {}".format((counter/numFs), head[0]))
 	
		looked += [head[0]]

	
	avg = []
	sets = []
	for i in range(2, 57, 2):
		dist = i / 1000
		for j in range(3, 162, 2):
			xx = '_EPS_%s_Samp_%s'%(dist,j)
			sets += [xx]
			counter = 0
			numFs = 0
			for obj in files:
				if xx in obj:
					counter += getlines(obj)
					numFs += 1
			avg += [(counter/numFs)]

	avgCount = pd.DataFrame()
	avgCount["AVG"] = avg
	avgCount["SEARCHED"] = sets
	#avgCount.sort_values(by=['AVG'], ascending=False)
	avgCount.to_csv("avg_changes_inData.csv", sep='\t', index = False) 

	xxx = []
	for i in range(len(avg)-1):
		xxx += [avg[i] - avg[i+1]]

	print("AVERAGE CAHNGE PER TEST {}".format(sum(xxx)/len(xxx)))
			#print("Average Change in clusters {} for the files {}".format((counter/numFs), xx))
			


			
