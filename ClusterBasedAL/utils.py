import torch
import random
import numpy as np

def random_sampler(selected_indices,n = 50000,sample_size = 2000) :
  random.seed(42)
  torch.manual_seed(42)
  np.random.seed(42)
  all_indices = list(range(n))

  all_indices = list(set(all_indices) -  set(selected_indices))

  sample_indices = random.sample(all_indices, sample_size)

  return sample_indices
