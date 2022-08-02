# Need to give this script the path name
for item in $1*.csv; do
	echo $item
#	python3 dbscan.py $item $2
	module load python/3.6.3; clusterize -m 25G -n 1 -p 20 -l 100:00:00 -o -c "python3 dbscan.py $item $2"
done
