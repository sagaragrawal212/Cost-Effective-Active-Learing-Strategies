# Cost-Effective-Active-Learing-Strategies.

## Abstract : 

Active learning, a paradigm in machine learning, involves iteratively selecting informative samples for annotation to enhance model performance with minimal labeled data. This research explores innovative methods for achieving cost-effective active learning. We propose two distinct approaches to optimize the annotation process while maintaining model efficacy. The first approach optimizes annotation expenses by leveraging high-confidence samples from the model's predictions in the current iteration as pseudo-labels for training in the next iteration, minimizing the need for extensive oracle feedback. The second method employs an unsupervised technique, utilizing Principal Component Analysis (PCA) for dimensionality reduction and K-means clustering to identify informative regions in the feature space. Samples closer to cluster centroids are then selectively annotated. In this research paper we delve deeper into these two methods and explore the pro's and cons for the same.


## Methods :

### Method 1 :  Cost-Effective Active Learning (CEAL)
This approach involves initially selecting a random batch of unlabeled data, which is then annotated with the assistance of a human oracle. Subsequently, a pertained Deep Neural Network (Alexnet) is trained using this annotated data. The trained model is then employed to make predictions on unlabeled data points.  The K most informative samples are identified using methods like least confidence, margin sampling and entropy and these are annotated with the oracle. Additionally, samples with high confidence (as determined by a threshold) have their predicted labels treated as pseudo labels, which are incorporated into the labeled set. The model is then updated through further training.

Steps:
This algorithm follows these key steps:
1. Take random sample of data points from unlabeled
dataset.
2. Annotate these sample using human oracle.
3. Train a classifier model on this labeled data using pretrained Alexnet.
4. Use this classifier to predict on remaining unlabeled
data.
5. Add the high confidence samples from these unlabeled data, whose entropy is smaller than the threshold
and their predicted label as pseudo label into labelled
dataset.
6. Take K most informative sample by taking K samples
with highest entropy.
7. Obtain labels for these K data points from human oracle, and add these K labeled data points to labelled
dataset.
8. Go to Step 3, update model by training it on these new
labeled datasets and repeat iteratively until the desired
level of accuracy is achieved or a termination criterion
is met.


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

