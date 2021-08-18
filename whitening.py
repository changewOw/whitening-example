"""
https://learndataa.com/2020/09/15/data-preprocessing-whitening-or-sphering-in-python/
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

# Create data
x = np.array([[1, 2, 3, 4, 5],  # Feature-1: Height
              [11, 12, 13, 14, 15]])  # Feature-2: Weight
x = x.T
print('x.shape:', x.shape)


# Center data
# By subtracting mean for each feature
xc = x - np.mean(x, axis=0)
print('xc.shape:', xc.shape, '\n')



# Calculate covariance matrix
xcov = np.cov(xc, rowvar=False, bias=True)
print('Covariance matrix: \n', xcov, '\n')



# Calculate Eigenvalues and Eigenvectors
w, v = linalg.eig(xcov)
# Note: Use w.real.round(4) to (1) remove 'j' notation to real, (2) round to '4' significant digits
print("Eigenvalues:\n", w.real.round(4), '\n')
print("Eigenvectors:\n", v, '\n')


# Calculate inverse square root of Eigenvalues
# Optional: Add '.1e5' to avoid division errors if needed
# Create a diagonal matrix
diagw = np.diag(1/(w**0.5)) # or np.diag(1/((w+.1e-5)**0.5))
diagw = diagw.real.round(4) #convert to real and round off
print("Diagonal matrix for inverse square root of Eigenvalues:\n", diagw, '\n')



# Whitening transform using PCA (Principal Component Analysis)
wpca = np.dot(np.dot(diagw, v.T), xc.T).T

# Whitening transform using ZCA (Zero Component Analysis)
wzca = np.dot(np.dot(np.dot(v, diagw), v.T), xc.T).T


# Plot zero-centered, rotated and whitened data
plt.scatter(xc[:,0], xc[:,1], s=150, label='centered', alpha=0.5)
# plt.scatter(xrot[0,:], xrot[1,:], s=150, label='rotated')
plt.scatter(wpca[:,0], wpca[:,1], s=150, marker='+', label='wpca')
plt.scatter(wzca[:,0], wzca[:,1], s=150, marker='x', label='wzca')
plt.xlabel('Height', fontsize=16)
plt.ylabel('Weight', fontsize=16)
plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)
plt.legend()
plt.tight_layout()
plt.savefig('whiten_1.png', dpi=300)






