from os import listdir
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

def extract(directory):
	files = listdir(directory)

	files = [f for f in files if "o.out" in f]

	scores = {}
	for ff in files:
		reading = open(directory+'/'+ff, 'r')
		tup = ()
		score = -2
		for line in reading:

			if "Parameters --> " in line:
				xx = line.split(' ')
				epsidx = xx.index('eps:')+1
				sampInd = xx.index('samples:')+1
				tup = (float(xx[epsidx]),int(xx[sampInd]))
			if "Score: " in line.strip():
				xx = line.split(' ')
				scoreIdx = xx.index("Score:") + 1
				score = float(xx[scoreIdx])
		if score != -2 and tup != ():
			scores[tup] = score

	silhouette = []
	eps = []
	min_samples = []
	for i in range(2, 57, 2):
		dist = i / 1000
		eps += [i]
		row = []
		for j in range(3, 162, 2):
			min_samples += [j]
			tup = (dist, j)
			row += [scores[tup]]
		silhouette += [row]

	return silhouette,eps,min_samples

def plotHeatMap(silhouette,eps,min_samples,f):
	fig, ax = plt.subplots(figsize=(15,15))
	silhouette = np.array(silhouette)

	heatMap = ax.imshow(silhouette)

        # Get the colorbar      
	cbar = ax.figure.colorbar(heatMap, ax=ax)
	cbarlabel = ""
	cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

	ax.set_xticks(np.arange(silhouette.shape[1]))
	ax.set_yticks(np.arange(silhouette.shape[0]))

	ax.set_xticklabels(min_samples)
	ax.set_yticklabels(eps)

	ax.set_xlabel("min_sabels")
	ax.set_ylabel("distance")
	ax.set_title('Grid Search Silhouette Score')

	# Let the horizontal axes labeling appear on bottom.
	ax.tick_params(top=False, bottom=True,
		labeltop=False, labelbottom=True)

	# Rotate the tick labels and set their alignment.
	#plt.setp(ax.get_xticklabels(), rotation=-90, ha="right",
	#       rotation_mode="anchor")
	plt.setp(ax.get_xticklabels(), rotation=-90)

	plt.savefig("%s.png"%(f))
	plt.close()

if __name__ == '__main__':
	
	directory = sys.argv[1]
	silhouette,eps,min_samples = extract(directory)

	plotHeatMap(silhouette,eps,min_samples, sys.argv[1])	
	
	d2 = sys.argv[2]
	silhouette1,eps1,min_samples1 = extract(d2)
	plotHeatMap(silhouette1,eps1,min_samples1, sys.argv[2])

	d3 =sys.argv[3]
	silhouette2,eps2,min_samples2 = extract(d3)

	plotHeatMap(silhouette2,eps2,min_samples2, sys.argv[3])

	d4 = sys.argv[4]
	silhouette3,eps3,min_samples3 = extract(d4)
	plotHeatMap(silhouette3,eps3,min_samples3, sys.argv[4])
	
