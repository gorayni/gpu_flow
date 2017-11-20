#!/bin/bash

baseFolder=$1
folderList=$(find $baseFolder -type d)
outputFolder=$2

for subFolder in $folderList 
do
	echo "Processing sub folfer $subFolder"
	./build/compute_flow -i "$subFolder" -o $outputFolder
done


