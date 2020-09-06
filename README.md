# Dataset Reuse Indicators

This repo provides data and subsequent model for the paper: 

Koesten, Laura and Vougiouklis, Pavlos and Simperl, Elena and Groth, Paul, **Dataset Reuse: Translating Principles to Practice.** Available at SSRN: https://ssrn.com/abstract=3589836 or http://dx.doi.org/10.2139/ssrn.3589836

This includes:

* Data for all github repos containing datasets (download_github_dataset.sh).
    * This contains a python pickle file. It is a list of each repo. A repo is represented as a hash that contains metadata about each repo containing a dataset. To unpickle use:
        ```
        file = open("dataset.pickle", 'rb')
        obj = pickle.load(file, encoding='latin1')
        obj[0] # gets the first repo in the list
        ``` 
* Data used for training models (download_processed_datasets.sh).
    * Code to work with this data is in the source code directory.
* The source code for model training (under reuse_predictor). Our model uses pytorch.

The shell scripts above are designed to make the data easy to access with this repo. The data can also be downloaded from Zenodo:

Koesten, Laura, Vougiouklis, Pavlos, Groth, Paul, & Simperl, Elena. (2020). Dataset Reuse Indicators Datasets (Version 1.0) [Data set]. Zenodo. http://doi.org/10.5281/zenodo.4015955

For more information contact: [Laura Koesten](https://laurakoesten.github.io)
