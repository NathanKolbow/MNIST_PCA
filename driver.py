# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 14:09:15 2020

@author: Nathan
"""

import pca
from scipy.linalg import eigh
import numpy as np


if __name__ == '__main__':
    data = pca.load_and_center_dataset("mnist.npy")
    cov = pca.get_covariance(data)
    
    #lam, eig = pca.get_eig(cov, 2)
    lam, eig = pca.get_eig_perc(cov, 0.001)
    
    image = data[3]
    projection = pca.project_image(image, eig)
    
    pca.display_image(image, projection)