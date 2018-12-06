# NeuralNetwork

Data for this project is too heavy to store on GitHub. 
Take Data from blackboard and copy directory into root of project.
Name of data directory should be 'pr_data'.

RankScore can be improved by applying K-Means clustering with Mahanalobis distance. 
The reason is because the spread of data within each class is approximately Gaussian. 
Some distributions are slightly skewed, therefore we apply a Logarithmic transform on the data,
such that Y = Log(X + 1), so that each class follows more closely a normal distribution. 
Adding 1 to X avoids mapping points between 0 and 1 to extreme values, e.g. lim(x->0) Log(x) -> -infinity. 
We can then consider reducing the dimensions of the data using linear PCA. 
Linear PCA will insure that the distribution remains gaussian, which is desirable when applying
Mahanalobis distance for clustering. However, keep in mind that an empirical rule for clustering
strongly suggests keeping a high dimensional data, as more dimensions makes separating data
generally easier. Consider applying linear PCA after clustering.
