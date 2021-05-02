#!/bin/sh
# Just for reproducibility
echo "Getting Data from UCI"
cd data
curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data
curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.names
curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/autos/misc
echo "Download complete"
