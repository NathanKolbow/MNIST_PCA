# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 14:09:15 2020

@author: Nathan
"""

import pca
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


def display_multiple_images(orig, projs, titles):
    # generates and displays the relevant plot
    fig, axs = plt.subplots(ncols=len(projs)+1)
    a1 = axs[0].imshow(orig.reshape((28, 28)), aspect='equal', cmap='gray')
    axs[0].set_title("Original")
    
    for i in range(len(projs)):
        a = axs[i+1].imshow(projs[i].reshape((28, 28)), aspect='equal', cmap='gray')
        axs[i+1].set_title(titles[i])
        
    plt.show()


if __name__ == '__main__':
    data = pca.load_and_center_dataset("mnist.npy")
    cov = pca.get_covariance(data)
    
    _, eig10 = pca.get_eig(cov, 10)
    _, eig25 = pca.get_eig(cov, 25)
    _, eig100 = pca.get_eig(cov, 100)
    _, eig250 = pca.get_eig(cov, 250)
    _, eig500 = pca.get_eig(cov, 500)
    
    image = data[782]
    projs = (pca.project_image(image, eig500),
             pca.project_image(image, eig250),
             pca.project_image(image, eig100),
             pca.project_image(image, eig25),
             pca.project_image(image, eig10))
    titles = ("n_components=500",
              "n_components=250",
              "n_components=100",
              "n_components=25",
              "n_components=10")
    
    display_multiple_images(image, projs, titles)