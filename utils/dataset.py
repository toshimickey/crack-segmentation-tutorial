import glob, os
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import json

class make_datapath_list():
  def __init__(self, nfold):
    # Load and shuffle image and annotation paths
    img_file_path = sorted(glob.glob('data/Train/images/*'))
    anno_file_path = sorted(glob.glob('data/Train/labels/*'))
    combined = list(zip(img_file_path, anno_file_path))
    random.shuffle(combined)
    img_file_path, anno_file_path = zip(*combined)

    # Split into folds
    self.nfold = nfold
    self.folds = []
    fold_size = len(img_file_path) // self.nfold
    for i in range(nfold):
        start = i * fold_size
        end = start + fold_size if i != nfold - 1 else len(img_file_path)
        self.folds.append((img_file_path[start:end], anno_file_path[start:end]))

    # Load test data
    img_file_path2 = sorted(glob.glob('data/Test/images/*'))
    anno_file_path2 = sorted(glob.glob('data/Test/labels/*'))
    self.test_file_path = img_file_path2
    self.test_anno_file_path = anno_file_path2

  def get_train_val_lists(self, fold_index):
      # Initialize empty lists for training and validation paths
      train_img_files = []
      train_anno_files = []
      val_img_files = []
      val_anno_files = []

      # Append data to train or val lists
      for i, (img_paths, anno_paths) in enumerate(self.folds):
          if i == fold_index:
              val_img_files.extend(img_paths)
              val_anno_files.extend(anno_paths)
          else:
              train_img_files.extend(img_paths)
              train_anno_files.extend(anno_paths)

      return (train_img_files, train_anno_files), (val_img_files, val_anno_files)

  def get_test_lists(self):
      return self.test_file_path, self.test_anno_file_path


class CrackDataset(Dataset):
    def __init__(self, img_list, label_list, transform=None):
        self.img_list = img_list
        self.label_list = label_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        label_path = self.label_list[idx]

        image = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('L')

        if self.transform:
            image, label = self.transform(image, label)

        return image, label
      

class CrackTransform():
  def __init__(self, crop_size):
      self.crop_size = crop_size

  def __call__(self, image, label):
    image = transforms.Resize((self.crop_size, self.crop_size))(image)
    label = transforms.Resize((self.crop_size, self.crop_size))(label)

    if random.random() > 0.5:
      image = transforms.functional.hflip(image)
      label = transforms.functional.hflip(label)

    if random.random() > 0.5:
      image = transforms.functional.vflip(image)
      label = transforms.functional.vflip(label)

    image = transforms.ToTensor()(image)
    label = transforms.ToTensor()(label)
    label = torch.where(label > 0.5, torch.tensor(1), torch.tensor(0)) 

    return image, label
  
