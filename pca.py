# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 11:33:17 2020

@author: Nathan
"""

from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


def load_and_center_dataset(filename):
    # Load and reshape the data to the proper shape
    data = np.load(filename).reshape((2000, 784))
    
    # Center the data; this is necessary for PCA
    return data - data.mean(axis=1).repeat(784).reshape((2000, 784))


def get_covariance(dataset):
    cov = np.zeros((784, 784))
    for i in range(len(dataset[0])):
        cov += np.dot(dataset[i].reshape(1, -1).transpose(), dataset[i].reshape(1, -1))
    
    return (1 / 1999) * cov


def get_eig(S, m):
    lam, v = eigh(S)
    return np.diag(np.flip(lam)[0:m]), np.flip(v)[0:m]


def get_eig_perc(S, perc):
    total = np.sum(S)
    
    valid = np.empty(0)
    for lam in S:
        if lam / total >= perc:
            valid = np.append(valid, lam)
            
    return valid


def project_image(image, U):
    projection = np.empty(784)
    for j in range(len(U)):
        projection += U[j] * image * U[j].transpose()
    return projection


def display_image(orig, proj):
    pass


if __name__ == '__main__':
    data = load_and_center_dataset("mnist.npy")
    #cov = get_covariance(data)
    cov = np.cov(data.transpose())
    lam, eig = eigh(cov)
    # NOTES: get_eig is retruning the wrong eigenvectors and get_covariance is incorrectly calculating covariance :)
    image = data[12]
    projection = project_image(image, eig)
    
    fig, axs = plt.subplots(ncols=2)
    axs[0].imshow(image.reshape((28, 28)))
    axs[1].imshow(projection.reshape((28, 28)))
    
    
    