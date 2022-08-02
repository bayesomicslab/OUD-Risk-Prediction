<?php

// Ensure that the four needed arguments is there
if ($argc != 5){
	echo "Four Arguments Requiered! Latitude, Longitude, API key, and file output name.\n";
	exit();
}

// Variables for latitude, longitude, and API key
$lat = $argv[1];
$long = $argv[2];
$key = $argv[3];
$fileName = $argv[4];

// Access the the API

// This line will acces the geocoding api 
//$url = sprintf('https://maps.googleapis.com/maps/api/geocode/json?latlng=%f,%f&key=%s',$lat,$long,$key);

// Accessing the places API to save output direct data to a file on the command line
$url = sprintf('https://maps.googleapis.com/maps/api/place/nearbysearch/json?location=%f,%f&radius=25&key=%s',$lat,$long,$key);
#$url = sprintf('https://maps.googleapis.com/maps/api/geocode/json?latlng=%s,%s&key=%s',$lat,$long,$key);
$ch = curl_init( $url );
$response = curl_exec( $ch );
curl_close($ch);
?>
