'''
Script designed to makePOST request to the google places API for every line in a file.

File name and API Key is command line input
'''
import sys
import os.path
import requests
import tempfile
import pandas as pd

'''
Function to make the URL post to the API
'''
def makePost(latitude,longitude,apiKey):
	# Make a post to the geocode API from google
	#url = 'https://maps.googleapis.com/maps/api/geocode/json?latlng=%f,%f&key=%s'%(lat,lng,key)
        
	# Make a post to the places API from google
	url = 'https://maps.googleapis.com/maps/api/place/nearbysearch/json?location=%f,%f&radius=25&key=%s'%(latitude,longitude,apiKey)
	response = requests.post(url = url)
	
	return response.text

'''
Generate a random file name
'''
def getFileName(idn):

	tf = tempfile.NamedTemporaryFile()      
	name = tf.name.split('/',2)
	outName = name[-1]

	out = str(idn) + '.' + outName
	return out


if __name__ == '__main__':
	
	# arg1 = fname, arg2 = API key, arg3 output path
	if len(sys.argv) != 4:
		print("File name,API Key, and outut path requiered to run this script")
		exit()
	
	fname = sys.argv[1]
	key = sys.argv[2]
	outDir = sys.argv[3]

	# get identity
	pathSplit = fname.split('/')
	identity = pathSplit[-1].split('_')
	identity = identity[0]

	# Parse the filie column 1 is not cluster number
	of = pd.read_csv(fname,sep=",")
	of.columns = ['cluster', 'latitude','longitude']
	
	for index, row in of.iterrows():
			
		clus, lat, lng = row.cluster, row.latitude, row.longitude	# Check the format of the files
		line1 = 'Cluster: {} latitude: {} longitude: {}\n'.format(clus, lat, lng)
		
		out = getFileName(clus)
			
		# Validate it is a proper float
		try:
			lat = float(lat)
			lng = float(lng)
		except:
			print("Make sure the latitude and longitude are float values!")
			continue

		response = makePost(lat,lng,key)
		
		# Check for a valid directory
		isdir = os.path.isdir("%s%s"%(outDir,identity))
		if  not isdir:
			os.mkdir(outDir + str(identity))		
	
		# Output the response to the POST request	
		outOpen = open("%s%s/"%(outDir,identity) + out,'w+')
		outOpen.write(line1)
		outOpen.write(response)
	
