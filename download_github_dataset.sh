#!/bin/bash

# Downloads and uncompresses our GitHub dataset.
mkdir -p data/github_dataset
cd data/github_dataset
wget -O github_dataset.zip https://www.dropbox.com/s/y2v6xrz154rt6oy/github_dataset.zip?dl=1
unzip -o github_dataset.zip
echo All required files have been downloaded and un-compressed successfully.