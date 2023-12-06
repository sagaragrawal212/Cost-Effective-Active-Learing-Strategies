
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from sklearn.cluster import KMeans
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torch.utils.data.sampler import SubsetRandomSampler
import random
import pickle

from kmeans import KMeansClusterer
from alexnet import AlexNet
from utils import random_sampler

import argparse

parser = argparse.ArgumentParser(description = """
                      max_run : maximum number of model runs
                      top_k : test Dataset size
                      dev_size : dev Dataset Size
                      model_type : name of model
                      """)
parser.add_argument('-runs',
            '--max_run',
            type = int,
            default = 2,
            help = 'maximum number of model runs')

parser.add_argument('-top_k',
            '--sample_size_per_cluster',
            type = int,
            default = 50,
            help = 'number of samples to be selected from each cluster')

parser.add_argument('-k',
            '--clusters',
            type = int,
            default = 10,
            help = 'Number of clusters')

parser.add_argument('-b',
            '--batch_size',
            type = int,
            default = 64,
            help = 'Batch size for training models')

parser.add_argument('-e',
            '--epochs',
            type = int,
            default = 40,
            help = 'Number of epochs for model training')

parser.add_argument('-method',
            '--sample_strategy',
            type = str,
            default = 'kmeans',
            help = 'Stratefy to select samples "random" or "kmeans"')

parser.add_argument('-d',
            '--dataset',
            type = str,
            default = 'cifar100',
            help = 'Dataset Name')

parser.add_argument('-n',
            '--n_classes',
            type = int,
            default = 100,
            help ='Number of classes')

parser.add_argument('-cf',
            '--cluster_flag',
            type = int,
            default = 1,
            help ='To do Clustering or not')

args = parser.parse_args()


if __name__ == "__main__" :
  random.seed(42)
  torch.manual_seed(42)
  np.random.seed(42)


  ## Parameters
  max_runs = args.max_run
  top_k = args.sample_size_per_cluster
  clusters = args.clusters
  batch_size = args.batch_size
  epochs = args.epochs
  method = args.sample_strategy
  dataset = args.dataset
  n_classes = args.n_classes
  cluster_flag = args.cluster_flag

  selected_indices = []
  acc_list = []


  if method == "random" :
    print("Starting Random Sampling Based Active Learning : ")
  elif method == "kmeans" :
    print("Starting Clustering Based Active Learning : ")

  # Step 1: Load the CIFAR-100 dataset
  transform = transforms.Compose([transforms.Resize(224),  # ViT model expects 224x224 images
                                  transforms.ToTensor()])

  if dataset == "cifar100" :
    cifar_train = datasets.CIFAR100(root="./data", train=True, transform=transform, download=True)
    cifar_val = datasets.CIFAR100(root="./data", train=False, transform=transform, download=True)

    cifar_train1 = datasets.CIFAR100(root="./data", train=True, transform=None, download=True)
    train_data_features = cifar_train1.data.reshape(cifar_train.data.shape[0], -1)
    train_data_labels = cifar_train1.targets

  elif dataset == "cifar10" :
    ## Initialization
    cifar_train = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
    cifar_val = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)

    cifar_train1 = datasets.CIFAR100(root="./data", train=True, transform=None, download=True)
    train_data_features = cifar_train1.data.reshape(cifar_train.data.shape[0], -1)
    train_data_labels = cifar_train1.targets

  if (method == "kmeans") and (cluster_flag == 1) :
    print('Dimensionality Reduction started ...')
    pca = PCA(n_components=0.99)
    train_data_embed = pca.fit_transform(train_data_features)
    labels = train_data_labels

    ## Clustering Step
    print('Clustering Started ...')
    clusterer = KMeansClusterer(train_data_embed, k=clusters)
    clusterer.fit()

    ## Saving the Cluster object
    with open("/content/drive/MyDrive/Project Proposal/kmeans_pca_k20_c100.pkl", "wb") as f:
      pickle.dump(clusterer, f)
  elif (method == "kmeans") and (cluster_flag == 0) :
    with open("/content/drive/MyDrive/Project Proposal/kmeans_pca_k20_c100.pkl", "rb") as f:
      clusterer = pickle.load(f)


  transform = transforms.Compose([
      transforms.Resize(224),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  if method == "kmeans" :
    train_indices = clusterer.get_top_indices(top_k)
  elif method == "random" :
    train_indices = random_sampler(selected_indices,n = 50000,sample_size = top_k*clusters)
    selected_indices.extend(train_indices)

  val_indices = list(range(len(cifar_val)))


  train_sampler = SubsetRandomSampler(train_indices)
  valid_sampler = SubsetRandomSampler(val_indices)

  train_dataloader = torch.utils.data.DataLoader(cifar_train, batch_size=batch_size,
                                    sampler=train_sampler, num_workers=4)
  val_dataloader = torch.utils.data.DataLoader(cifar_val, batch_size=batch_size,
                                      sampler=valid_sampler,num_workers=4)


  # Create the model
  model = AlexNet(n_classes=n_classes, device=None)

  for run in range(max_runs) :

    # Initialize the model
    print('Intialize training the model')

    model.train(epochs=epochs, train_loader=train_dataloader, valid_loader=None)
    PATH = '/content/drive/MyDrive/Project Proposal/model_dataset_{dataset}_train_size_{len(train_dataloader.dataset)}_method_{method}.pkl'
    torch.save(model, PATH)

    # Evaluate model on dtest
    acc = model.evaluate(test_loader=val_dataloader)
    acc_list.append(acc)
    print('====> Accuracy: {} '.format(acc))

    print("Updating the Training Data : ")

    if method == "kmeans" :
      new_train_indices = clusterer.get_top_indices(top_k)
    elif method == "random" :
      new_train_indices = random_sampler(selected_indices,n = 50000,sample_size = top_k*clusters)
      selected_indices.extend(new_train_indices)

    train_dataloader.sampler.indices.extend(new_train_indices)

    print("All accuracies : ")
    print(acc_list)
