#!/bin/bash

# Downloads and uncompresses all the required files for training our reuse predictor.
mkdir -p data/processed_dataset
cd data/processed_dataset
wget -O processed_dataset.zip https://www.dropbox.com/s/z27f7sojdzi7czo/processed_dataset.zip?dl=1
unzip -o processed_dataset.zip
echo All required files have been downloaded and un-compressed successfully.