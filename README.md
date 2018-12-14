# NeuralNetwork

### Setup
Data for this project is too heavy to store on GitHub. 
Take Data from blackboard and copy directory into root of project.
Name of data directory should be 'pr_data'.

### Gradient Ascent Algorithm & Iterative Projection
Value of step size 'alpha' is crucial. Don't modify unless confident.

### Testing
If you do not have scientific mode in Pycharm, then code will pause when a plot
is displayed. To resume code, close plot windows.

### Configurations
A YAML file is provided in './cfgs' folder. File determines basic configurations
with which to run code. More explanations are given in file.

### Source
Source is made up of a 'main.py', and local modules 'train.py', 'model.py', 'test.py',
'functions.py'. 

'main.py' is the file you want to run to load and process the data.

'train.py' is a short module that only contains functions to partition the data for 
validation and training.

'model.py' contains the functions for training: gradient ascent & iterative projection
and pca (KMeans is a function imported from open source library 'sklearn').

'test.py' contains the functions to partition the data into query and gallery sets and
testing for rank retrieval. Camera ID is taken into account.

### Numpy files
'./src' folder contains '.npy' files that we urge you NOT to remove. 

'train_mean.npy' is the mean of all points used for training after flattening.

'metric_log_file.npy' contains trained metric after Cholesky decomposition, which in this 
case is just the square root of a diagonal matrix.

'cluster_means_file.npy' contains the cluster centers of the baseline data.

'cluster_means_transformed_file.npy' contains the cluster centers of the transformed data.