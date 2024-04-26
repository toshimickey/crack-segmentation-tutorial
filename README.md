# Crack Segmentation Tutorial　[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/toshimickey/crack-segmentation-tutorial/blob/master/main.ipynb)

This repository contains a Python script to train a cracks detection model. The model can be easily executed from Google Colab.

### Dataset

- The `data` folder contains default crack images. However, feel free to add your own crack images for training. Place your images in the `data` folder, and if necessary, divide them into `Train` and `Test` folders. Files placed in the `Train` folder will undergo cross-validation with the specified fold size, eliminating the need for separate validation data.

### Training Weights

- The `weights` folder stores the trained weights for each fold during the training process.

