from subprocess import call
import sys

fileName = sys.argv[1]
for i in range(2, 57, 2):
	dist = i / 1000
	for j in range(3, 162, 2):
		call(["module load python/3.6.3; clusterize -m 15G -n 1 -p 10 -l 100:00:00 -o -c \"python3 gridSearch.py %s %s %s\""%(fileName,dist,j)], shell=True)

