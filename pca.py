#############################
# pca.py
# Nathan Kolbow, Fall 2020
# CS 540
#############################
from numpy.linalg import eig
import numpy as np
import matplotlib.pyplot as plt


# Loads the pre-arranged NumPy data and centers it so that each
# pixel's mean is 0
def load_and_center_dataset(filename):
    # Load and reshape the data to the proper shape
    data = np.load(filename).reshape((2000, 784))
    
    # Center the data; this is necessary for PCA
    return data - data.mean(axis=1).repeat(784).reshape((2000, 784))


# Gets the covariance matrix of the dataset
def get_covariance(dataset):
    return np.dot(dataset.T, dataset) / (dataset.shape[1] - 1)


# Get the m-largest eigenvalues and eigenvectors of the given matrix S
def get_eig(S, m):
    lam, v = eig(S)
    return np.real(np.diag(lam[0:m])), np.real(v[:, 0:m])


# Gets all of the eigenvalues and eigenvectors that contribute to at least
# <perc>*100% of the variance in images
def get_eig_perc(S, perc):
    lam, v = eig(S)
    
    total = np.sum(lam)
    valid = np.empty(0)
    for i in lam:
        if i / total >= perc:
            valid = np.append(valid, i)
    
    return np.real(np.diag(valid)), np.real(v[np.isin(lam, valid)])


# Project <image> onto the eigenvectors supplied in <U>
def project_image(image, U):
    projection = np.zeros((784, 1))
    
    # project the image onto the eigenspace U
    for i in range(len(U[0])):
        eig = U[:, i].reshape(-1, 1)
        dot = eig.T.dot(image.reshape(-1, 1))
        projection += dot * eig
        
    return projection
        

# Given two 1x784 length arrays orig and proj, creates a side-by-side plot
# displaying the origin 28x28 image and the same image after being after
# being projected across the eigenspace
def display_image(orig, proj):
    # generates and displays the relevant plot
    fig, axs = plt.subplots(ncols=2)
    a1 = axs[0].imshow(orig.reshape((28, 28)), aspect='equal', cmap='gray')
    axs[0].set_title("Original")
    a2 = axs[1].imshow(proj.reshape((28, 28)), aspect='equal', cmap='gray')
    axs[1].set_title("Projection")
    
    fig.colorbar(a1, ax=axs[0])
    fig.colorbar(a2, ax=axs[1])
    plt.show()
    
    