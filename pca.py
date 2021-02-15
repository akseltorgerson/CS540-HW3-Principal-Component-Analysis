from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

# Loads dataset from filename and returns it centered about the origin
def load_and_center_dataset(filename):
    dataset = np.load(filename)
    mean = np.mean(dataset, axis = 0)
    center = dataset - mean
    return center

# Returns covariance of the dataset
def get_covariance(dataset):
    transpose = np.transpose(dataset)
    covariance = (np.dot(transpose, dataset)) / (len(dataset) - 1)
    return covariance
    
# Returns the M largest eigen values and their vectors
def get_eig(S, m):
    val, vec = eigh(S, subset_by_index=[len(S) - m, len(S) - 1])
    val = np.flip(val)
    vec = np.fliplr(vec)
    val_matrix = np.diag(val)
    return val_matrix, vec[:, :len(val_matrix)]

# Get all Eigenvals/vecs with more than a certain % of variance
def get_eig_perc(S, perc):
    val, vec = eigh(S)
    val = np.flip(val)
    vec = np.fliplr(vec)
    greater = []
    for value in val:
        calc_perc = value / (np.sum(val))
        if calc_perc > perc:
            greater.append(value)
    val_matrix = np.diag(greater)
    return val_matrix, vec[:, :len(val_matrix)]
    
# Gets the projected imagine from a given imagine using the eigen vectors
def project_image(img, U):
    proj_img = np.dot(U, np.transpose(U))
    proj_img = np.dot(proj_img, img)
    return proj_img
    
# Display both the original and projected images
def display_image(orig, proj):
    orig_reshape = np.reshape(orig, (32, 32), order='F')
    proj_reshape = np.reshape(proj, (32, 32), order='F')
    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (9, 3))
    axes[0].set_title('Original')
    axes[1].set_title('Projection')
    cb1 = axes[0].imshow(orig_reshape, aspect = 'equal', cmap = 'viridis')
    cb2 = axes[1].imshow(proj_reshape, aspect = 'equal', cmap = 'viridis')
    fig.colorbar(cb1, ax=axes[0])
    fig.colorbar(cb2, ax=axes[1])
    plt.show()
    return
    
