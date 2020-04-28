# Dataset Reuse Indicators

This repo provides data and subsequent model for the paper "Dataset Reuse: Translating Principles to Practice". 

This includes:

* Data for all github repos containing datasets (download_github_dataset.sh).
    * This contains a python pickle file. It is a list of each repo. A repo is represented as a hash that contains metadata about each repo containing a dataset. To unpickle use:
        ```
        file = open("dataset.pickle", 'rb')
        obj = pickle.load(file, encoding='latin1')
        ``` 
* Data used for training models (download_processed_datasets.sh).
    * Code to work with this data is in the source code directory.
* The source code for model training (under reuse_predictor). Our model uses pytorch.

For more information contact: [Laura Koesten](https://laurakoesten.github.io)
