# Introduction

This repository contains my experiments on the [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales/data) dataset
using the [fastai](https://github.com/fastai/fastai) library.

# Usage

1. Install Docker

2. Navigate to this project's root directory

3. Build and run this project's Dockerfile by executing:
    
    ```bash
    docker build -t fastai . && \
    docker run --runtime=nvidia \
               --name fastai \
               -p 8888:8888 \
               --dns 8.8.8.8 \
               fastai jupyter notebook --ip 0.0.0.0 --allow-root
    ```
      This will start a jupyter server on port 8888.
      
4. Open a web browser and navigate to the URL displayed in the previous step.

5. Navigate to the [notebooks](/notebooks) folder. This folder contains:
    * [01_rossmann_data_clean](/notebooks/01_rossman_data_clean.ipynb): Download the training and test set
    * [02_EDA](/notebooks/02_EDA.ipynb): Some exploratory data analysis on the dataset
    * [03_modelling](/notebooks/03_modelling.ipynb): Models I've tried so far
    
    # Contributing
    Feel free to raise a pull request for any questions, comments, feedback, etc.