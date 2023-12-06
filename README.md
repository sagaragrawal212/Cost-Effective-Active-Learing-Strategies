# Cost-Effective-Active-Learing-Strategies.

## Abstract : 

Active learning, a paradigm in machine learning, involves iteratively selecting informative samples for annotation to enhance model performance with minimal labeled data. This research explores innovative methods for achieving cost-effective active learning. We propose two distinct approaches to optimize the annotation process while maintaining model efficacy. The first approach optimizes annotation expenses by leveraging high-confidence samples from the model's predictions in the current iteration as pseudo-labels for training in the next iteration, minimizing the need for extensive oracle feedback. The second method employs an unsupervised technique, utilizing Principal Component Analysis (PCA) for dimensionality reduction and K-means clustering to identify informative regions in the feature space. Samples closer to cluster centroids are then selectively annotated. In this research paper we delve deeper into these two methods and explore the pro's and cons for the same.


## Methods :

### Method 1 : 

### Method 2 Clustering Based Active Learning Algotrithms : 

In scenarios where a substantial amount of data remains unlabeled, and there is a budget available for acquiring annotations, the challenge lies in selecting the initial set of data points for annotation. In this context, we propose the utilization of a clustering-based active learning methodologies for the strategic selection of data points to be labeled by an annotator.

#### PCA Clustering Based Active Learning method : 

Steps : 

- Reduce the dimensionality of data by applying Principal Component Analysis
- Apply K-means clustering to the low dimension data 
- Select sample of data points from each cluster that are closest to the cluster centroid from each cluster.
- Obtain annotations from the annotator for the selected data points.
- Train the machine learning model using the annotated data.
- Evaluate the model's performance.
- Reiterate the process by sampling the next data points closest to the cluster centroid from each cluster.
- Go to Step 4 and repeat iteratively until the desired level of accuracy is achieved or a termination criterion is met.


#### Code Execution Command : 

**Baseline Random sampling based AL** : 

```
python main.py -runs 3 -method 'random' -d 'cifar10' -n 10 -top_k 2 -k 250
```

**PCA K-Means sampling based AL** : 

```
python main.py -runs 3 -d 'cifar10' -n 10 -top_k 2 -k 250  -method 'kmeans' -cf 1
```

### Results : 

![image info](https://github.com/sagaragrawal212/Cost-Effective-Active-Learing-Strategies/blob/main/ClusterBasedAL/results.png)

### References : 

