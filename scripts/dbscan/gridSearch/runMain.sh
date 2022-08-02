# Need to give this script the path name
for item in $1*.csv; do
	echo $item
	python3 gridSearch.py $item $2
done
