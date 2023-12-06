#imports
import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from typing import Optional, Callable
import cv2
import tarfile
import os
import glob

if not os.path.exists('data/cifar10.tgz'):
  # Dowload the dataset into a "data" folder
  dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
  download_url(dataset_url, 'data')

  # Extract archive file 
  with tarfile.open('data/cifar10.tgz', 'r:gz') as tar:
      tar.extractall(path='data/')

  # Print class names in cifar10 dataset
  data_dir = 'data/cifar10/'
  print("class names in cifar10 dataset : ", os.listdir(data_dir + '/train'))

  #Renaming class folder of train set to numerical values (0 to 9)
  train_dir = 'data/cifar10/train/*'
  for i, each_dir in enumerate(glob.glob(train_dir)):
    os.rename(each_dir, train_dir[:-1] + str(i+1))

  #Renaming class folder of test to numerical values (0 to 9)
  test_dir = 'data/cifar10/test/*'
  for i, each_dir in enumerate(glob.glob(test_dir)):
    os.rename(each_dir, test_dir[:-1] + str(i+1))
    
else:
  data_dir = 'data/cifar10/'

print("class tags in cifar10 dataset : ", os.listdir(data_dir + '/train'))
print("Total images in train set : ",len(glob.glob(data_dir + '/train' +'/*/*')))
print("Total images in test set : ", len(glob.glob(data_dir + '/test' +'/*/*')))


### Encapsulate Cifar10 torch.utils.data.Dataset
class Cifar10(Dataset):
    """
    Parameters
    root_dir :  Path to the dataset directory.
    transform :  A transform function that takes the original image and  return a transformed version.
    Attributes 
    data : list of images files names
    labels :  list of integers (labels)
    """

    def __init__(self, root_dir: str,
                 transform: Optional[Callable] = None):

        self.root_dir = os.path.expanduser(root_dir)
        self.transform = transform
        self.data = []
        self.labels = []
        self._classes = 10

        # load data and labels
        for cat in range(0, self._classes):
          cat_dir = glob.glob(os.path.join(self.root_dir, str(cat+1)))[0]
          for img_file in glob.glob(os.path.join(cat_dir, '*.png')):
            self.data.append(img_file)
            self.labels.append(cat)

    def __getitem__(self, idx: int) -> dict:
        img, label = self.data[idx], self.labels[idx]
        label = torch.tensor(label,dtype = torch.long)
        img = cv2.imread(img)
        img = img[:, :, ::-1]
        img = self.img_normalize(img).transpose(2, 0, 1)
        sample = {'image': img, 'label': label}

        if self.transform:
            sample['image'] = self.transform(torch.tensor(sample['image']))

        return sample

    def __len__(self):

        return len(self.data)

    @staticmethod
    def img_normalize(img):
        img = (img / 255.0)

        return img
